/*
 * application_twr_pdoa_tag.c
 *
 *  Created on: July 14, 2022
 *      Author: Tobias Margiani
 */

//#include "applications.h"
#include <example_selection.h>

#include <stdio.h>
#include <string.h>

//#include "main.h"
#include <deca_regs.h>
#include <deca_spi.h>
#include <port.h>
#include "uart_stdio.h"
#include <wiringPi.h> // for getting time in milliseconds
#include <assert.h> // for static_assert
#include <math.h> // for calculating aoa

#include "application_config.h"
#include "shared_functions.h"


#if defined(APP_TWR_PDOA)

#define DEVICE_MAX_NUM 10 // number of expected devices to communicate with
#define FORCE_ANCHOR 1 // Force the device to be an anchor and never enter tag mode
#define FORCE_TAG 0 // Force the device to be a tag

// antenna delays for calibration
#define TX_ANT_DLY (16385-60)
#define RX_ANT_DLY (16385-60)

extern dwt_txconfig_t txconfig_options;

static void tx_done_cb(const dwt_cb_data_t *cb_data);
static void rx_ok_cb(const dwt_cb_data_t *cb_data);
static void rx_err_cb(const dwt_cb_data_t *cb_data);

volatile uint8_t rx_done = 0;  /* Flag to indicate a new frame was received from the interrupt */
volatile uint16_t new_frame_length = 0;
volatile uint8_t tx_done = 0;
volatile unsigned int last_recieve_time;

char print_buffer[128];

twr_base_frame_t sync_frame = {
		{ 0x41, 0x88 },	/* Frame Control: data frame, short addresses */
		0,				/* Sequence number */
		{ 'X', 'X' },	/* PAN ID */
		{ 'A', 'A' },	/* Destination address */
		{ 'T', 'T' },	/* Source address */
		0x20,			/* Function code: 0x20 ranging initiation */
		/* According to ISO/IEC 24730-62:2013 this should be sent by the anchor and end with a short
		 * address temporarily assigned to the tag. We invert the whole tag/anchor process to compute
		 * the ranging on the tag where we have access to the AoA estimation and skip the two bytes
		 * for short address in this message for simplicity. */
};

twr_base_frame_t response_frame = {
		{ 0x41, 0x88 },	/* Frame Control: data frame, short addresses */
		0,				/* Sequence number */
		{ 'X', 'X' },	/* PAN ID */
		{ 'A', 'A' },	/* Destination address */
		{ 'T', 'T' },	/* Source address */
		0x10,			/* Function code: 0x10 activity control */
		/* According to ISO/IEC 24730-62:2013 this frame should have another 3 octets added for an
		 * option code and parameters we skip this here fore simplicity. */
};

twr_base_frame_t poll_frame = {
		{ 0x41, 0x88 },	/* Frame Control: data frame, short addresses */
		0,				/* Sequence number */
		{ 'X', 'X' },	/* PAN ID */
		{ 'T', 'T' },	/* Destination address */
		{ 'A', 'A' },	/* Source address */
		0x21,			/* Function code: 0x21 ranging poll */
};

twr_final_frame_t final_frame = {
		{ 0x41, 0x88 },		/* Frame Control: data frame, short addresses */
		0,					/* Sequence number */
		{ 'X', 'X' },		/* PAN ID */
		{ 'T', 'T' },		/* Destination address */
		{ 'A', 'A' },		/* Source address */
		0x23,				/* Function code: 0x22 ranging final with embedded timestamp */
		{ 0, 0, 0, 0, 0 },	/* Time from TX of poll to RX of response frame (i.e. Tround1) */
		{ 0, 0, 0, 0, 0 },	/* Time from RX of response to TX of final frame (i.e. Treply2) */
		0,					/* PDoA measured by the anchor (pdoa_tx) */
		/* According to ISO/IEC 24730-62:2013 the thre timestamps at the end should be only 32-bits each
		 * but then we would just discard values and loose accuracy. */
};

static const size_t max_frame_length = sizeof(twr_final_frame_t) + 2;

uint64_t rx_timestamp_poll = 0;
uint64_t tx_timestamp_response = 0;
uint64_t rx_timestamp_final = 0;
uint64_t tx_timestamp_poll = 0;
uint64_t rx_timestamp_response = 0;
uint64_t tx_timestamp_final = 0;

int16_t pdoa_rx = 0;
int16_t pdoa_tx = 0;
uint8_t tdoa_rx[5];

uint8_t next_sequence_number = 0;

// unique chip ID of the decawace chip. Used for identification
const uint16_t CHIPID_ADDR = 0x06;
uint32_t device_id;

enum state_t {
	TWR_SYNC_STATE_TAG,
	TWR_POLL_RESPONSE_STATE_TAG,
	TWR_FINAL_STATE_TAG,
	TWR_SYNC_STATE_ANC,
	TWR_POLL_RESPONSE_STATE_ANC,
	TWR_FINAL_STATE_ANC,
	TWR_ERROR_TAG,
	TWR_ERROR_ANC,
};

enum state_t state = TWR_SYNC_STATE_ANC;
uint8_t tag_mode = 0; // keeps track of if we are in tag (1) or anchor (0) mode

/* timeout before the ranging exchange will be abandoned and restarted */
static const uint64_t round_tx_delay = 900llu*US_TO_DWT_TIME;  // reply time (0.7ms) now 10 ms
static const unsigned int tag_sync_timeout = 10; // (10 ms) 100 ms
static const unsigned int anc_resp_timeout = 10; // slightly smaller than sync timeout
static const unsigned int min_tx_timeout = 5; // min timout value
static const unsigned int avg_tx_timeout = 20; // (5 ms) 100 ms, Should be at least four times that of round_tx_delay 
unsigned int tx_timeout = min_tx_timeout + avg_tx_timeout/2; // the timeout, before reverting to anchor

void transmit_rx_diagnostics(float current_rotation, int16_t pdoa_rx, int16_t pdoa_tx, uint8_t * tdoa);
void print_hex(const uint8_t *bytes, size_t length);
int8_t checkTO(unsigned int * last_time, unsigned int timeout);
/**
 * Application entry point.
 */
int application_twr_pdoa_tag(void)
{
    /*		------------------------------------------------- THE FRANKENCODE BEGINS -------------------------------------------------	 */
	stdio_init();
	printf("DW3000 TEST TWR Tag\n");

    /* Configure SPI rate, DW IC supports up to 38 MHz */
    port_set_dw_ic_spi_fastrate();

    /* Reset DW IC */
    reset_DWIC(); /* Target specific drive of RSTn line into DW IC low for a period. */

    Sleep(20); // Time needed for DW3000 to start up (transition from INIT_RC to IDLE_RC, or could wait for SPIRDY event)

    while (!dwt_checkidlerc()) /* Need to make sure DW IC is in IDLE_RC before proceeding */
    { };

    if (dwt_initialise(DWT_DW_INIT) == DWT_ERROR)
    {
    	printf("INIT FAILED\n");
        while (1) { };
    }

    printf("INITIALIZED\n");

    /* Enabling LEDs here for debug so that for each RX-enable the D2 LED will flash on DW3000 red eval-shield boards. */
    dwt_setleds(DWT_LEDS_ENABLE | DWT_LEDS_INIT_BLINK);

    /* Configure DW IC. */
    if(dwt_configure(&config)) /* if the dwt_configure returns DWT_ERROR either the PLL or RX calibration has failed the host should reset the device */
    {
    	printf("CONFIG FAILED\n");
        while (1)
        { };
    }

	// configure transmit power and pulse delay
	dwt_configuretxrf(&txconfig_options);

	// apply antenna delays
	dwt_setrxantennadelay(RX_ANT_DLY);
    dwt_settxantennadelay(TX_ANT_DLY);

    printf("CONFIGURED\n");

    /* Register RX call-back. */
    dwt_setcallbacks(tx_done_cb, rx_ok_cb, rx_err_cb, rx_err_cb, NULL, NULL);

    /* Enable wanted interrupts (TX confirmation, RX good frames, RX timeouts and RX errors). */
    dwt_setinterrupt(SYS_ENABLE_LO_TXFRS_ENABLE_BIT_MASK | SYS_ENABLE_LO_RXFCG_ENABLE_BIT_MASK | SYS_ENABLE_LO_RXFTO_ENABLE_BIT_MASK |
            SYS_ENABLE_LO_RXPTO_ENABLE_BIT_MASK | SYS_ENABLE_LO_RXPHE_ENABLE_BIT_MASK | SYS_ENABLE_LO_RXFCE_ENABLE_BIT_MASK |
            SYS_ENABLE_LO_RXFSL_ENABLE_BIT_MASK | SYS_ENABLE_LO_RXSTO_ENABLE_BIT_MASK, 0, DWT_ENABLE_INT);


    /*Clearing the SPI ready interrupt*/
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RCINIT_BIT_MASK | SYS_STATUS_SPIRDY_BIT_MASK);

    /* Install DW IC IRQ handler. */
    port_set_dwic_isr(dwt_isr);

	/* Activate reception immediately (as we start out as an anchor). */
    dwt_rxenable(DWT_START_RX_IMMEDIATE);

    uint8_t timestamp_buffer[5];
    uint8_t rx_buffer[max_frame_length];
    twr_base_frame_t *rx_frame_pointer;
    twr_final_frame_t *rx_final_frame_pointer;
    int16_t sts_quality_index;
    unsigned int last_sync_time = millis(); // replaced HAL_GetTick();
	last_recieve_time = millis();

    float current_rotation = 0;
    uint16_t twr_count = 0;

    /*Get unique chip ID from OTP memory as device identification*/
	dwt_otpread(CHIPID_ADDR, &device_id, 1);
	uint8_t my_ID[2]; // id of tag 
	my_ID[0] = device_id >> 24 & 0xFF;
	my_ID[1] = device_id >> 16 & 0xFF;
	print_hex(my_ID, 2);
	/*This hacky code is a way to make sure that the devices own ID dosent appear with the other IDs*/
	uint8_t known_IDs[] = {0x96, 0xC2, 0x16, 0xC2};
	uint8_t device_num = 1; // current number of known devices
	uint8_t device_crt = 0; // the current index of device
	uint8_t your_ID[2]; // expected address of incoming message
	uint8_t your_ID_list[device_num*2];
	int temp_idx = 0;
	for (int i=0; i < (device_num+1); i++){
		if (memcmp(my_ID, &known_IDs[i*2], 2) != 0)
		{
			print_hex(&known_IDs[2*i], 2);
			your_ID_list[temp_idx++] = known_IDs[2*i];
			your_ID_list[temp_idx++] = known_IDs[2*i+1];
		}
	}

	if (FORCE_ANCHOR) {
		printf("Device set as anchor.\n");
	}
	if (FORCE_TAG) {
		state = TWR_SYNC_STATE_TAG;
		dwt_forcetrxoff(); // this is important - receivemode needs to be disabled to tx
		printf("Device set as tag.\n");
	}
	printf("Wait 3s before starting...\n");
    Sleep(3000);

	while (1)
	{
		switch (state) {
		
		/*		------------------------------------------------- ANCHOR CODE -------------------------------------------------	 */
		
		case TWR_SYNC_STATE_ANC:
			/* If device is forced to be an anchor only, then never change over to tag */
			if (!FORCE_ANCHOR){
				if ((millis() - last_recieve_time) > tx_timeout) {
					/* If it is time to transmit: */
					dwt_forcetrxoff();
					/* Calculate random timeout time, centered around the average*/
					tx_timeout = min_tx_timeout + (rand() % (avg_tx_timeout+1));
					//last_sync_time = millis();
					printf("Changing into tag\n");
					tag_mode = 1;
					state = TWR_SYNC_STATE_TAG; // We become a tag
					tx_timestamp_poll = 0;
					rx_timestamp_response = 0;
					tx_timestamp_final = 0;
					tx_done = 0;
					rx_done = 0;
					// continue?
				}
			}
			/* Wait for sync frame (1/4) */
			if (rx_done)
			{
				rx_done = 0;

				if (new_frame_length != sizeof(twr_base_frame_t)+2) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { /* >= 0 good STS, < 0 bad STS */
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* We assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x20) {  /* ranging init */
					printf("RX ERR: wrong frame (expected sync)\n");
					state = TWR_ERROR_ANC;
					continue;
				}
				/* code for adding new devices dynamically. Commented out since we assume known addresses
				// Now we know it is a sync frame, we can check if we know the src address:
				uint8_t device_known = 0;
				for (int i=0; i < device_num; i++){
					if (memcmp(&your_ID_list[i*2], rx_frame_pointer->src_address, 2) == 0){
						// we know the device. Ignore it
						device_known = 1;
						break;
					}
				}
				if (!device_known){
					printf("UNKNOWN SRC ADDRESS. Adding following to list:\n");
					if (device_num < DEVICE_MAX_NUM){
						memcpy(&your_ID_list[device_num*2], rx_frame_pointer->src_address, 2);
						print_hex(&your_ID_list[device_num*2], 2);
						device_num++; 
					} else {
						printf("Device list full. Ignoring");
					}	
				}
				*/				
				
				if (memcmp(my_ID, rx_frame_pointer->dst_address, 2) != 0) {
					printf("RX ERR: wrong dest address on sync frame\n");
	
					state = TWR_ERROR_ANC;
					continue;
				}

				printf("RX: Sync frame\n");

				/* Set the expected source to that of the incoming messages source, to ignore all other messages*/
				memcpy(your_ID, rx_frame_pointer->src_address, 2);

				/* Initialize the sequence number for this ranging exchange */
				next_sequence_number = rx_frame_pointer->sequence_number + 1;

				/* Send poll frame (2/4) */
				state = TWR_POLL_RESPONSE_STATE_ANC; /* Set early to ensure tx done interrupt arrives in new state */
				// set dest and src:
				memcpy(poll_frame.src_address, my_ID, 2);
				memcpy(poll_frame.dst_address, your_ID, 2);
				poll_frame.sequence_number = next_sequence_number++;
				dwt_writetxdata(sizeof(poll_frame), (uint8_t *)&poll_frame, 0);
				dwt_writetxfctrl(sizeof(poll_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */
				int r = dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
				if (r != DWT_SUCCESS) {
					printf("TX ERR: could not send poll frame\n");
					state = TWR_ERROR_ANC;
					continue;
				}
			}
			break;
		case TWR_POLL_RESPONSE_STATE_ANC:
			if (tx_done == 1) {
				tx_done = 2;
				printf("TX: Poll frame\n");
				dwt_readtxtimestamp(timestamp_buffer);
				tx_timestamp_poll = decode_40bit_timestamp(timestamp_buffer);
				last_sync_time = millis();
			}

			/* check timeout for response frame*/
			if (rx_done == 0 && tx_done == 2){
				if (checkTO(&last_sync_time, anc_resp_timeout)){
					printf("Anchor response timeout\n");
					state = TWR_ERROR_ANC;
					continue;
				}
			}

			/* Wait for response frame (3/4) */
			if (rx_done == 1) {
				rx_done = 0; /* reset */

				if (new_frame_length != sizeof(twr_base_frame_t)+2) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { /* >= 0 good STS, < 0 bad STS */
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* We assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x10) { /* response */
					printf("RX ERR: wrong frame (expected response)\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				if (rx_frame_pointer->sequence_number != next_sequence_number) {
					printf("RX ERR: wrong sequence number\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				if (memcmp(my_ID, rx_frame_pointer->dst_address, 2) != 0) {
					printf("RX ERR: wrong dest address on response frame\n");
					state = TWR_ERROR_ANC;
					continue;
				}
				
				if (memcmp(your_ID, rx_frame_pointer->src_address, 2) != 0) {
					printf("RX ERR: wrong souce address on response frame\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				printf("RX: Response frame\n");
				dwt_readrxtimestamp(timestamp_buffer);
				rx_timestamp_response = decode_40bit_timestamp(timestamp_buffer);

				/* get the PDoA of the response frame (AoD)*/
				pdoa_tx = dwt_readpdoa(); 

				/* Accept frame and continue ranging */
				next_sequence_number++;
				rx_done = 2;
			}

			if ((tx_done == 2) && (rx_done == 2)) {
				tx_done = 0;
				rx_done = 0;

				/* Send final frame (4/4) */
				// set dest and src:
				memcpy(final_frame.src_address, my_ID, 2);
				memcpy(final_frame.dst_address, your_ID, 2);
				final_frame.sequence_number = next_sequence_number++;

				tx_timestamp_final = rx_timestamp_response + round_tx_delay + TX_ANT_DLY; // inclue antenna delay in final timestamp

				uint64_t Tround1 = rx_timestamp_response - tx_timestamp_poll;
				uint64_t Treply2 = tx_timestamp_final - rx_timestamp_response;

				final_frame.poll_resp_round_time[0] = (uint8_t)Tround1;
				final_frame.poll_resp_round_time[1] = (uint8_t)(Tround1 >> 8);
				final_frame.poll_resp_round_time[2] = (uint8_t)(Tround1 >> 16);
				final_frame.poll_resp_round_time[3] = (uint8_t)(Tround1 >> 32);

				final_frame.resp_final_reply_time[0] = (uint8_t)Treply2;
				final_frame.resp_final_reply_time[1] = (uint8_t)(Treply2 >> 8);
				final_frame.resp_final_reply_time[2] = (uint8_t)(Treply2 >> 16);
				final_frame.resp_final_reply_time[3] = (uint8_t)(Treply2 >> 32);

				final_frame.pdoa_tx = pdoa_tx;

				dwt_writetxdata(sizeof(final_frame), (uint8_t *)&final_frame, 0);
				dwt_writetxfctrl(sizeof(final_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */

				/* Start transmission at the time we embedded into the message */
				state = TWR_FINAL_STATE_ANC; /* Set early to ensure tx done interrupt arrives in new state */
				
				uint32_t final_tx_time = (rx_timestamp_response + round_tx_delay) >> 8;
				dwt_setdelayedtrxtime(final_tx_time);
				int r = dwt_starttx(DWT_START_RX_DELAYED | DWT_RESPONSE_EXPECTED);
				if (r != DWT_SUCCESS) {
					printf("TX ERR: delayed send time missed");
					state = TWR_ERROR_ANC;
					continue;
				}
			}
			break;
		case TWR_FINAL_STATE_ANC:
			if (tx_done == 1) {
				tx_done = 0;
				printf("TX: Final frame\n");
				state = TWR_SYNC_STATE_ANC;
			}
			break;
		case TWR_ERROR_ANC:
			printf("Anchor error -> reset\n");
			state = TWR_SYNC_STATE_ANC;
			Sleep(2);
			dwt_rxenable(DWT_START_RX_IMMEDIATE);
			break;

		/*		------------------------------------------------- TAG CODE -------------------------------------------------	 */

		case TWR_SYNC_STATE_TAG:
			/* Send sync frame (1/4) */
			// First check if we have looped through all destination addresses
			if (device_crt >= device_num) {
				if (device_crt == 0){
					// Special case: there's currently no known devices. So just broadcast some crap, and get to know someone!
					printf("Sending random msg: Notice me!\n");
					memcpy(your_ID, your_ID_list, 2);
					device_crt++;
				} else {
					if (!FORCE_TAG){
						// restart the receiver and turn off transmitter
						dwt_forcetrxoff();
						dwt_rxenable(DWT_START_RX_IMMEDIATE);
						last_recieve_time = millis();
						printf("No devices left: Changing into anchor\n");
						state = TWR_SYNC_STATE_ANC;
						device_crt = 0;
						tag_mode = 0;
						rx_timestamp_poll = 0;
						tx_timestamp_response = 0;
						rx_timestamp_final = 0;
						tx_done = 0;
						rx_done = 0;
						continue;
					} else {
						// just keep looping. if forced as tag
						device_crt = 0;
						continue;
					}
				}
			} else {
				// If there is still devices left, then access these 
				printf("Next address\n");
				memcpy(your_ID, &your_ID_list[device_crt*2], 2);
				device_crt++;
			}
			//last_sync_time = millis(); // replaced HAL_GetTick() 
			sync_frame.sequence_number = next_sequence_number++;
			// Set the destination and source
			memcpy(sync_frame.dst_address, your_ID, 2);
			memcpy(sync_frame.src_address, my_ID, 2);
			dwt_writetxdata(sizeof(sync_frame), (uint8_t *)&sync_frame, 0);
			dwt_writetxfctrl(sizeof(sync_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */

			state = TWR_POLL_RESPONSE_STATE_TAG; /* Set early to ensure tx done interrupt arrives in new state */
			int r = dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
			if (r != DWT_SUCCESS) {
				state = TWR_ERROR_TAG;
				printf("TX ERR: could not send sync frame\n");
				continue;
			}
			break;
		case TWR_POLL_RESPONSE_STATE_TAG:
			if (tx_done == 1) {
				tx_done = 2;
				printf("TX: Sync frame\n");
				last_sync_time = millis();
			}

			/* Poll frame timeout*/
			if (rx_done == 0 && tx_done == 2){
				if (checkTO(&last_sync_time, tag_sync_timeout)){
					printf("Tag Poll timeout\n");
					state = TWR_ERROR_TAG;
					continue;
				}
			}

			/* Wait for poll frame (2/4) */
			if (rx_done == 1) {
				rx_done = 0; /* reset */

				if (new_frame_length != sizeof(twr_base_frame_t)+2) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { /* >= 0 good STS, < 0 bad STS */
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* We assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x21) { /* poll */
					printf("RX ERR: wrong frame (expected poll)\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				if (rx_frame_pointer->sequence_number != next_sequence_number) {
					printf("RX ERR: wrong sequence number\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				if (memcmp(my_ID, rx_frame_pointer->dst_address, 2) != 0) {
					printf("RX ERR: wrong dest address on poll frame\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				if (memcmp(your_ID, rx_frame_pointer->src_address, 2) != 0) {
					printf("RX ERR: wrong source address on poll frame\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				printf("RX: Poll frame\n");

				dwt_readrxtimestamp(timestamp_buffer);
				rx_timestamp_poll = decode_40bit_timestamp(timestamp_buffer);

				/* Accept frame and continue ranging */
				next_sequence_number++;
				rx_done = 2;
			}

			if ((tx_done == 2) && (rx_done == 2)) {
				tx_done = 0;
				rx_done = 0;

				/* Send response frame (3/4) */
				// Set the destination and source
				memcpy(response_frame.dst_address, your_ID, 2);
				memcpy(response_frame.src_address, my_ID, 2);
				response_frame.sequence_number = next_sequence_number++;
				dwt_writetxdata(sizeof(response_frame), (uint8_t *)&response_frame, 0);
				dwt_writetxfctrl(sizeof(response_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */

				// Send response after a fixed delay
				state = TWR_FINAL_STATE_TAG; /* Set early to ensure tx done interrupt arrives in new state */

				uint32_t resp_tx_time = (rx_timestamp_poll + round_tx_delay) >> 8;
				dwt_setdelayedtrxtime(resp_tx_time);
				int r = dwt_starttx(DWT_START_TX_DELAYED | DWT_RESPONSE_EXPECTED);
				if (r != DWT_SUCCESS) {
					printf("TX ERR: delayed send time missed\n");
					state = TWR_ERROR_TAG;
					continue;
				}
			}
			break;
		case TWR_FINAL_STATE_TAG:
			if (tx_done == 1) {
				tx_done = 2;
				printf("TX: Response frame\n");
				dwt_readtxtimestamp(timestamp_buffer);
				tx_timestamp_response = decode_40bit_timestamp(timestamp_buffer);
				last_sync_time = millis();
			}
			
			/* check timeout for response frame*/
			if (rx_done == 0 && tx_done == 2){
				if (checkTO(&last_sync_time, tag_sync_timeout)){
					printf("Tag final timeout\n");
					state = TWR_ERROR_TAG;
					continue;
				}
			}

			/* Wait for final frame (4/4) */
			if (rx_done == 1) {
				rx_done = 0; /* reset */

				if (new_frame_length != sizeof(twr_final_frame_t)+2) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { /* >= 0 good STS, < 0 bad STS */
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* For simplicity we assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x23) { /* final */
					printf("RX ERR: wrong frame (expected final)\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				if (rx_frame_pointer->sequence_number != next_sequence_number) {
					printf("RX ERR: wrong sequence number\n");
					state = TWR_ERROR_TAG;
					continue;
				}
				
				if (memcmp(my_ID, rx_frame_pointer->dst_address, 2) != 0) {
					printf("RX ERR: wrong dest address on final frame\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				if (memcmp(your_ID, rx_frame_pointer->src_address, 2) != 0) {
					printf("RX ERR: wrong source address on final frame\n");
					state = TWR_ERROR_TAG;
					continue;
				}

				printf("RX: Final frame\n");

				dwt_readrxtimestamp(timestamp_buffer);
				rx_timestamp_final = decode_40bit_timestamp(timestamp_buffer);

				/* Marker for serial output parsing script*/
				snprintf(print_buffer, sizeof(print_buffer), "New Frame: poll: %u\n", next_sequence_number);
				printf(print_buffer);

				/* Transmit measurement data */
				current_rotation = getAngle(); // for debugging
				pdoa_rx = dwt_readpdoa();
				dwt_readtdoa(tdoa_rx); // the tdoa measurements are pretty much useless, since antennas are too close

				/* Accept frame continue with ranging */
				next_sequence_number++;
				rx_done = 2;
			}

			if ((tx_done == 2) && (rx_done == 2)) {
				rx_final_frame_pointer = (twr_final_frame_t *)rx_buffer;

				const uint64_t Treply1 = tx_timestamp_response - rx_timestamp_poll;
				const uint64_t Tround2 = rx_timestamp_final - tx_timestamp_response;

				const uint64_t Tround1 = decode_40bit_timestamp(rx_final_frame_pointer->poll_resp_round_time);
				const uint64_t Treply2 = decode_40bit_timestamp(rx_final_frame_pointer->resp_final_reply_time);

				const uint64_t subtraction = (Tround1*Tround2 - Treply1*Treply2);
				const uint64_t denominator = (Tround1 + Tround2 + Treply1 + Treply2);

				// timestamp resolution is approximately u=15.65ps => 1ns = 63.898*u
				// to get ns the division by 63.898 is approximated by an division by 64 using a bit shift
				const float tprop_ns = ((double)subtraction) / (denominator << 6);
				const uint32_t dist_mm = (uint32_t)(tprop_ns*299.792458);  // usint c = 299.7... mm/ns

				pdoa_tx = rx_final_frame_pointer->pdoa_tx;

				// Write to log CSV file
				csv_write_twr(Treply1, Treply2, Tround1, Tround2, dist_mm, twr_count, current_rotation);
				transmit_rx_diagnostics(current_rotation, pdoa_rx, pdoa_tx, tdoa_rx); // log true rotation, measured pdoa and tdoa

				/* Transmit human readable for debugging */
				snprintf(print_buffer, sizeof(print_buffer), "twr_count: %u, dist_m: %.2f\n", twr_count, ((float)dist_mm)/1000);
				printf(print_buffer);

				/* Rotate receiver */
				twr_count++;

				/* Begin next ranging exchange */
				tx_done = 0;
				rx_done = 0;
				state = TWR_SYNC_STATE_TAG;

				Sleep(100); // short pause after succesful exchange
			}
			break;
		case TWR_ERROR_TAG:
			dwt_forcetrxoff();  // make sure receiver is off after an error
			printf("Tag error -> reset\n");
			state = TWR_SYNC_STATE_TAG;
			Sleep(2);
		}
	}

    return DWT_SUCCESS;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn tx_done_cb()
 *
 * @brief Callback called after TX
 *
 * @param  cb_data  callback data
 *
 * @return  none
 */
static void tx_done_cb(const dwt_cb_data_t *cb_data)
{
	UNUSED(cb_data);
	tx_done = 1;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn rx_ok_cb()
 *
 * @brief Callback to process RX good frame events
 *
 * @param  cb_data  callback data
 *
 * @return  none
 */
static void rx_ok_cb(const dwt_cb_data_t *cb_data)
{
	rx_done = 1;
	new_frame_length = cb_data->datalength;
	last_recieve_time = millis();
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn rx_err_cb()
 *
 * @brief Callback to process RX error and timeout events
 *
 * @param  cb_data  callback data
 *
 * @return  none
 */
static void rx_err_cb(const dwt_cb_data_t *cb_data)
{
	UNUSED(cb_data);
	/* restart rx on error */
	dwt_forcetrxoff();
	dwt_rxenable(DWT_START_RX_IMMEDIATE);
}

int8_t checkTO(unsigned int * last_time, unsigned int timeout){
		/* check ranging timeout and restart ranging if necessary  */
		if ((millis() - *last_time) > timeout) { 
			//*last_time = millis();
			rx_timestamp_poll = 0; // maybe delete all this
			tx_timestamp_poll = 0;
			rx_timestamp_response = 0;
			tx_timestamp_response = 0;
			rx_timestamp_final = 0;
			tx_timestamp_final = 0;
			tx_done = 0;
			rx_done = 0;
			return 1;
		} else {
			return 0;
		}
}


void transmit_rx_diagnostics(float current_rotation, int16_t pdoa_rx, int16_t pdoa_tx, uint8_t * tdoa) {
	// Readable pdoa stuff
	float pdoa_read_rx = ((float)pdoa_rx / (1 << 11));
	float pdoa_read_tx = ((float)pdoa_tx / (1 << 11));

	// angle
	float eq_lamb = 4.6196;
	float eq_d = 2.31;
	float eq_aoa = asinf((pdoa_read_rx*eq_lamb)/(2*M_PI*eq_d)) * (180/M_PI);

	// Readabe tdoa stuff
	int64_t tdoa_read = ((uint64_t)tdoa[0]) \
						+ ((uint64_t)tdoa[1] << 8) \
						+ ((uint64_t)tdoa[2] << 16) \
						+ ((uint64_t)tdoa[3] << 24) \
						+ ((uint64_t)tdoa[4] << 32);
	if (tdoa[5] & 0x01){
		// negative signed number. Set all upper 24 bits to 1:
		tdoa_read = tdoa_read | 0xffffff0000000000;
	}

	snprintf(print_buffer, sizeof(print_buffer), "raw pdoa rx: %.6f, raw pdoa tx: %.6f, tdoa: %ld, aoa: %.6f, True angle: %.1f \n",  pdoa_read_rx, pdoa_read_tx, tdoa_read, eq_aoa, current_rotation);
	printf(print_buffer);

	csv_write_rx(pdoa_read_rx, tdoa_read, current_rotation);
}

void print_hex(const uint8_t *bytes, size_t length) {
	for (size_t i = 0; i < length; i++) {
		printf("%02X ", bytes[i]);
	}
	printf("\n");
}

#endif
