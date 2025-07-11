/*
 * application_twr_pdoa_tag.c
 *
 *  Created on: July 14, 2022
 *      Author: Tobias Margiani 
 *  Properly fixed and expanded on: July 1, 2025
 * 		Author: Marcus lari Jørgensen
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
#include <time.h>
#include <assert.h> // for static_assert
#include <math.h> // for calculating aoa

#include "application_config.h"
#include "shared_functions.h"


#if defined(APP_TWR_PDOA)

#define DEVICE_MAX_NUM 10 // number of expected devices to communicate with
#define FORCE_ANCHOR 0 // Force the device to be an anchor and never enter tag mode
#define FORCE_TAG 0 // Force the device to be a tag

// antenna delays for calibration
#define TX_ANT_DLY (16385)
#define RX_ANT_DLY (16385)

extern dwt_txconfig_t txconfig_options;

static void tx_done_cb(const dwt_cb_data_t *cb_data);
static void rx_ok_cb(const dwt_cb_data_t *cb_data);
static void rx_err_cb(const dwt_cb_data_t *cb_data);

volatile uint8_t rx_done = 0;  /* Flag to indicate a new frame was received from the interrupt */
volatile uint16_t new_frame_length = 0;
volatile uint8_t tx_done = 0;
volatile uint64_t last_recieve_time;

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
		{ 0, 0, 0, 0},		/* Time from TX of poll to RX of response frame (i.e. Tround1) */
		{ 0, 0, 0, 0},		/* Time from RX of response to TX of final frame (i.e. Treply2) */
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
uint8_t sync_sequence_number = 0;
float dist_sum = 0;
uint8_t init_mode = 0;


enum state_t {
	TWR_SYNC_STATE_TAG,
	TWR_WAIT_FOR_CLEAR_STATE_ANC,
	TWR_POLL_RESPONSE_STATE_TAG,
	TWR_FINAL_STATE_TAG,
	TWR_SYNC_STATE_ANC,
	TWR_SYNC_TX_STATE_TAG,
	TWR_POLL_RESPONSE_STATE_ANC,
	TWR_FINAL_STATE_ANC,
	TWR_ERROR_TAG,
	TWR_ERROR_ANC,
};

enum state_t state = TWR_SYNC_STATE_ANC;

/* timeout before the ranging exchange will be abandoned and restarted */
static const uint64_t round_delay_us = 1200; // reply time (1ms)
static const uint64_t round_tx_delay = round_delay_us*US_TO_DWT_TIME; // reply time in dut 
  			 uint64_t tag_sync_timeout = round_delay_us+500; //(1.5 ms) How much time before the tag stops looking for a response (us)
static const uint64_t anc_resp_timeout = round_delay_us+500; //(1.5 ms) How much time before the anchor stops looking for a response (us)
static const uint64_t min_poll_timeout = round_delay_us + 500; // min time to wait before transmitting poll
static const uint64_t max_poll_timeout = 10000; // 10 ms. Max time in us to wait before attempting to transmit poll. The average time will be max_poll_timeout/2 + min_poll_timeout
static const uint64_t responses_timeout = max_poll_timeout + min_poll_timeout + 500; // when the tag should stop waiting for responses. Should be smaller than min_tx_timeout, to avoid to simostanously tags. +500 to make sure it can catch late polls
static const uint64_t min_tx_timeout = responses_timeout+1000; // min timout value (us) - the minimum time a node at least has to attempt being an anchor. Should be larger than responses timeout to avoid two tags
static const uint64_t max_tx_timeout = 3000000; // (20 ms). Max timeout calue is this + min timeout. Adjust according to how many tags are active at once
			 uint64_t tx_timeout = min_tx_timeout + max_tx_timeout/2; // the initial timeout, before reverting to tag
			 uint64_t poll_timeout = 0; // The time to wait before attempting to transmit poll

void transmit_rx_diagnostics(uint64_t ts, uint16_t id, float current_rotation, int16_t pdoa_rx, int16_t pdoa_tx, uint8_t * tdoa);
void print_hex(const uint8_t *bytes, size_t length);
uint64_t get_time_us();

int8_t checkTO(uint64_t * last_time, uint64_t timeout);
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

    //uint8_t timestamp_buffer[5];
    uint8_t rx_buffer[max_frame_length];
    twr_base_frame_t *rx_frame_pointer;
    twr_final_frame_t *rx_final_frame_pointer;
    int16_t sts_quality_index;
    uint64_t last_sync_time = get_time_us(); 
	last_recieve_time = get_time_us();

    float current_rotation = 0;
    uint16_t twr_count = 0;

	/*Assign random ID to each device*/
	srand(time(NULL));
	uint8_t my_ID[2]; // ID of tag
	uint8_t your_ID[2]; // ID of destination
	my_ID[0] = rand() % 256;
	my_ID[1] = rand() % 256;
	print_hex(my_ID, 2);

	if (FORCE_ANCHOR) {
		printf("Device set as anchor.\n");
	}
	if (FORCE_TAG) {
		state = TWR_SYNC_STATE_TAG;
		dwt_forcetrxoff(); // this is important - receivemode needs to be disabled to tx
		tag_sync_timeout = 1000; // special case - we want to be patient when we are the only tag
		init_mode = 0;
		printf("Device set as tag.\n");
	}

	// Write device ID to CSV before starting
	uint16_t my_ID16 = (uint16_t)my_ID[0] + ((uint16_t)my_ID[1] << 8);
	csv_write_info(my_ID16, FORCE_ANCHOR, FORCE_TAG);

	printf("Wait 3s before starting...\n");
    Sleep(3000);

	uint64_t start_time = get_time_us();

	while (1)
	{
		switch (state) {
		
		/*		------------------------------------------------- ANCHOR CODE -------------------------------------------------	 */
		
		case TWR_SYNC_STATE_ANC:
			/* If device is forced to be an anchor only, then never change over to tag */
			if (!FORCE_ANCHOR){
				if ((get_time_us() - last_recieve_time) > tx_timeout) {
					/* If it is time to transmit: */
					dwt_forcetrxoff();
					printf("Changing into tag\n");
					state = TWR_SYNC_STATE_TAG; // We become a tag
					//tx_timestamp_poll = 0;
					//rx_timestamp_response = 0;
					//tx_timestamp_final = 0;
					tx_done = 0;
					rx_done = 0;
					init_mode = 1;
					continue;
				}
			}
			/* Wait for sync frame (1/4) */
			if (rx_done)
			{
				rx_done = 0;

				// some printf statements are commented out here, since errors are expected here, when other nodes do DS TWR after this node has finsihed ranging
				if (new_frame_length > max_frame_length) {
					//printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { // >= 0 good STS, < 0 bad STS 
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_ANC;
					continue;
				}

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* We assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x20) {  /* ranging init */
					//printf("RX ERR: wrong frame (expected sync but got x%02x)\n", rx_frame_pointer->twr_function_code);
					state = TWR_ERROR_ANC;
					continue;
				}

				state = TWR_WAIT_FOR_CLEAR_STATE_ANC;
				/*set the random poll response time*/
				poll_timeout = min_poll_timeout + (rand() % (max_poll_timeout+1));
				
				/* Set the expected source to that of the incoming messages source, to ignore all other messages*/
				memcpy(your_ID, rx_frame_pointer->src_address, 2);

				printf("RX: Sync frame from ");
				print_hex(your_ID, 2);

				/* Initialize the sequence number for this ranging exchange */
				next_sequence_number = rx_frame_pointer->sequence_number + 1;
				// set dest and src:
				memcpy(poll_frame.src_address, my_ID, 2);
				memcpy(poll_frame.dst_address, your_ID, 2);
				poll_frame.sequence_number = next_sequence_number++;
				dwt_writetxdata(sizeof(poll_frame), (uint8_t *)&poll_frame, 0);
				dwt_writetxfctrl(sizeof(poll_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */
				// start the reciver again to sense for reception
				dwt_rxenable(DWT_START_RX_IMMEDIATE);
			}
			break;
		case TWR_WAIT_FOR_CLEAR_STATE_ANC:
			
			// wait for clear airwaves before attempting to respond:
			// another timeout can be added here in case the airwaves never become clear, to avoid the node being stauck waiting to tarnsmit 
			if (rx_done==1){
				// We have received a message. Just discard it and restart the receiver
				printf("Detected channel activity. Waiting...\n");
				rx_done = 0;
				dwt_rxenable(DWT_START_RX_IMMEDIATE);
			} else if ((get_time_us() - last_recieve_time) > poll_timeout){
				// Clear! now transmit:
				/* Send poll frame (2/4) */
				printf("Airwaves are clear.Preparing to send poll...\n");
				// Turn off the receiver, or else we wont be able to transmit
				dwt_forcetrxoff();
				state = TWR_POLL_RESPONSE_STATE_ANC; /* Set early to ensure tx done interrupt arrives in new state */

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
				last_sync_time = get_time_us();
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

				if (new_frame_length > max_frame_length) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_ANC;
					continue;
				}
				
				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { // >= 0 good STS, < 0 bad STS 
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

				rx_timestamp_response = get_rx_timestamp_u64();
				tx_timestamp_poll = get_tx_timestamp_u64();

				// include antenna delay in final timestamp. Remember to remove lower 9 bits, 
				// since this is what setdelayedtrx does - if this is neglcted, you get some bad noise
				tx_timestamp_final = (((rx_timestamp_response + round_tx_delay) >> 9) << 9) + TX_ANT_DLY;

				// Cast every timestamp to 32 bit to avoid artifacts from clock wrapping
				uint32_t tx_timestamp_poll_32 = (uint32_t)tx_timestamp_poll;
				uint32_t rx_timestamp_response_32 = (uint32_t)rx_timestamp_response;
				uint32_t tx_timestamp_final_32 = (uint32_t)tx_timestamp_final;

				uint32_t Tround1 = rx_timestamp_response_32 - tx_timestamp_poll_32;
				uint32_t Treply2 = tx_timestamp_final_32 - rx_timestamp_response_32;

				final_frame.poll_resp_round_time[0] = (uint8_t)Tround1;
				final_frame.poll_resp_round_time[1] = (uint8_t)(Tround1 >> 8);
				final_frame.poll_resp_round_time[2] = (uint8_t)(Tround1 >> 16);
				final_frame.poll_resp_round_time[3] = (uint8_t)(Tround1 >> 24);

				final_frame.resp_final_reply_time[0] = (uint8_t)Treply2;
				final_frame.resp_final_reply_time[1] = (uint8_t)(Treply2 >> 8);
				final_frame.resp_final_reply_time[2] = (uint8_t)(Treply2 >> 16);
				final_frame.resp_final_reply_time[3] = (uint8_t)(Treply2 >> 24);

				final_frame.pdoa_tx = pdoa_tx;

				dwt_writetxdata(sizeof(final_frame), (uint8_t *)&final_frame, 0);
				dwt_writetxfctrl(sizeof(final_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */

				/* Start transmission at the time we embedded into the message */
				state = TWR_FINAL_STATE_ANC; /* Set early to ensure tx done interrupt arrives in new state */
				
				uint32_t final_tx_time = (rx_timestamp_response + round_tx_delay) >> 8;
				dwt_setdelayedtrxtime(final_tx_time);
				int r = dwt_starttx(DWT_START_RX_DELAYED | DWT_RESPONSE_EXPECTED);
				if (r != DWT_SUCCESS) {
					printf("TX ERR: delayed send time missed for final frame\n");
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
				// update timeout after afinished exchange - incase the anchor is stuck with a bad high timeout
				tx_timeout = min_tx_timeout + (rand() % (max_tx_timeout+1));
				last_recieve_time = get_time_us(); // might as well update the receive time afer succesful DS TWR
				// write to log that a succesful exchange was made
				uint64_t log_ts = get_time_us() - start_time; // get timestamp for measurement
				uint16_t your_ID16 = (uint16_t)your_ID[0] + ((uint16_t)your_ID[1] << 8); // convert id to 16 bit
				csv_write_id(log_ts, your_ID16);
			}
			break;
		case TWR_ERROR_ANC:
			//printf("Anchor error -> reset\n");
			state = TWR_SYNC_STATE_ANC;
			dwt_rxenable(DWT_START_RX_IMMEDIATE);
			break;

		/*		------------------------------------------------- TAG CODE -------------------------------------------------	 */

		case TWR_SYNC_STATE_TAG:
			/* Send sync frame (1/4) */
			sync_frame.sequence_number = next_sequence_number++; 
			/*Remember this sequence number, so we can reset back to it when initiating a ranging with a new anchor from this sync message*/
			sync_sequence_number = next_sequence_number; 
			// Set the source. Destination does not matter (Sync initiates ranging with all anchors)
			memcpy(sync_frame.src_address, my_ID, 2);
			dwt_writetxdata(sizeof(sync_frame), (uint8_t *)&sync_frame, 0);
			dwt_writetxfctrl(sizeof(sync_frame)+2, 0, 1); /* Zero offset in TX buffer, ranging. */

			state = TWR_SYNC_TX_STATE_TAG; /* Set early to ensure tx done interrupt arrives in new state */
			int r = dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
			if (r != DWT_SUCCESS) {
				state = TWR_ERROR_TAG;
				printf("TX ERR: could not send sync frame\n");
				// maybe try again if not forced as tag?
				continue;
			}
			break;
		case TWR_SYNC_TX_STATE_TAG:
			if (tx_done == 1) {
				state = TWR_POLL_RESPONSE_STATE_TAG;
				printf("TX: Sync frame\n");
				last_sync_time = get_time_us();
				tx_done = 0;
			}
			break;
		case TWR_POLL_RESPONSE_STATE_TAG:
			/* 	In this state, the tag awaits all the delayed responses from the anchors
				reacting to the initial SYNC state. If no reactions is detected, the tag
				reverts back to being an anchor */

			/* Poll frame timeout*/
			if (rx_done == 0){
				// change to anchor if no response
				if (get_time_us() - last_sync_time > responses_timeout){
					if (!FORCE_TAG){
						// restart the receiver and turn off transmitter
						dwt_forcetrxoff();
						dwt_rxenable(DWT_START_RX_IMMEDIATE);
						last_recieve_time = get_time_us();
						/* Calculate random timeout time, centered around the average*/
						tx_timeout = min_tx_timeout + (rand() % (max_tx_timeout+1));
						printf("No responses left: Changing into anchor\n");
						state = TWR_SYNC_STATE_ANC;
						//rx_timestamp_poll = 0;
						//tx_timestamp_response = 0;
						//rx_timestamp_final = 0;
						tx_done = 0;
						rx_done = 0;
						init_mode = 0;
						continue;
					} else {
						printf("Tag Poll timeout.\n");
						state = TWR_ERROR_TAG;
						continue;
					}
				}
			}

			/* Wait for poll frame (2/4) */
			if (rx_done == 1) {
				rx_done = 0; /* reset */
				next_sequence_number = sync_sequence_number; /* reset back to the original seqence number used for sync*/

				if (new_frame_length > max_frame_length) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_TAG;
					continue;
				}
				
				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { // >= 0 good STS, < 0 bad STS 
					printf("RX ERR: bad STS quality\n");
					state = TWR_ERROR_TAG;
					continue;
				} 

				dwt_readrxdata(rx_buffer, new_frame_length, 0);
				/* We assume this is a TWR frame, but not necessarily the right one */
				rx_frame_pointer = (twr_base_frame_t *)rx_buffer;

				if (rx_frame_pointer->twr_function_code != 0x21) { /* poll */
					printf("RX ERR: wrong frame (expected poll) but got x%02x\n", rx_frame_pointer->twr_function_code);
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

				// We got a poll from an random anchor - we want to finish the ranging with only this anchor
				memcpy(your_ID, rx_frame_pointer->src_address, 2);

				printf("RX: Poll frame\n");

				rx_timestamp_poll = get_rx_timestamp_u64();

				/* Accept frame and continue ranging */
				next_sequence_number++;
				rx_done = 2;
			}

			if (rx_done == 2) {
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
				last_sync_time = get_time_us();
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

				if (new_frame_length > max_frame_length) {
					printf("RX ERR: wrong frame length\n");
					state = TWR_ERROR_TAG;
					continue;
				}
				
				int sts_quality = dwt_readstsquality(&sts_quality_index);
				if (sts_quality < 0) { // >= 0 good STS, < 0 bad STS 
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
			
				/* Transmit measurement data */
				//current_rotation = getAngle(); // for debugging. Reanble for PDoA tests
				pdoa_rx = dwt_readpdoa();
				dwt_readtdoa(tdoa_rx); // the tdoa measurements are pretty much useless, since antennas are too close. This can be removed

				/* Accept frame continue with ranging */
				next_sequence_number++;
				rx_done = 2;
			}

			if ((tx_done == 2) && (rx_done == 2)) {
				rx_final_frame_pointer = (twr_final_frame_t *)rx_buffer;

				tx_timestamp_response = get_tx_timestamp_u64();
				rx_timestamp_final = get_rx_timestamp_u64();

				// convert to 32 bit
				uint32_t rx_timestamp_poll_32 = (uint32_t)rx_timestamp_poll;
				uint32_t tx_timestamp_response_32 = (uint32_t)tx_timestamp_response;
				uint32_t rx_timestamp_final_32 = (uint32_t)rx_timestamp_final;

				double Treply1 = (double)(tx_timestamp_response_32 - rx_timestamp_poll_32);
				double Tround2 = (double)(rx_timestamp_final_32 - tx_timestamp_response_32);

				double Tround1 = (double)(decode_32bit_timestamp(rx_final_frame_pointer->poll_resp_round_time));
				double Treply2 = (double)(decode_32bit_timestamp(rx_final_frame_pointer->resp_final_reply_time));

				double subtraction = (Tround1*Tround2 - Treply1*Treply2);
				double denominator = (Tround1 + Tround2 + Treply1 + Treply2);

				// timestamp resolution is approximately u=15.65ps => 1ns = 63.898*u
				// to get ns the division by 63.898 is approximated by an division by 64 using a bit shift
				double tprop_ns = (subtraction) / (denominator*63.898);
				uint64_t dist_mm = (uint64_t)(tprop_ns*299.792458);  // usint c = 299.7... mm/ns

				pdoa_tx = rx_final_frame_pointer->pdoa_tx;

				printf("Finished ranging with node with address ");
				print_hex(your_ID, 2);

				// Write to log CSV file
				uint64_t log_ts = get_time_us() - start_time; // get timestamp for measurement
				uint16_t your_ID16 = (uint16_t)your_ID[0] + ((uint16_t)your_ID[1] << 8); // convert id to 16 bit
				csv_write_twr2(log_ts, your_ID16, Treply1, Treply2, Tround1, Tround2, dist_mm, twr_count, current_rotation);
				transmit_rx_diagnostics(log_ts, your_ID16, current_rotation, pdoa_rx, pdoa_tx, tdoa_rx); // log true rotation, measured pdoa and tdoa

				/* Transmit human readable for debugging */
				dist_sum += dist_mm;
				float dist_mean = dist_sum/((twr_count+1)*1000); // mean distance in meters
				snprintf(print_buffer, sizeof(print_buffer), "twr_count: %u, dist_m: %.3f, mean dist: %.3f \n", twr_count, ((float)dist_mm)/1000, dist_mean);
				printf(print_buffer);

				/* Rotate receiver */
				twr_count++;

				/* Begin next ranging exchange. We go back to the response state and await te delayed responses from the other anchors */
				tx_done = 0;
				rx_done = 0;
				last_sync_time = get_time_us();
				state = TWR_POLL_RESPONSE_STATE_TAG;
				/*Enable receiver, in order to look for next poll*/
				dwt_rxenable(DWT_START_RX_IMMEDIATE);
				if (FORCE_TAG){Sleep(10);} // just to slow down the measurements a bit when running tag only
			}
			break;
		case TWR_ERROR_TAG:
			//printf("Tag error -> reset\n");
			if (FORCE_TAG){
				// If forced to be tag, send new sync
				dwt_forcetrxoff(); // reset, so we can send new message
				Sleep(10);
				state = TWR_SYNC_STATE_TAG;
			} else {
				// otherwise, wait for anchor response (dont reset trx here, as we want to stay in receive mode)
				dwt_rxenable(DWT_START_RX_IMMEDIATE);
				state = TWR_POLL_RESPONSE_STATE_TAG;
			}
			break;
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
	last_recieve_time = get_time_us();
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
	
	printf("RX error");
	if (init_mode) {
		// if the node is an initator
		state = TWR_ERROR_TAG;
	} else {
		// if the node is a responder
		state = TWR_ERROR_ANC;
	}
	
	/* restart rx on error */
	dwt_forcetrxoff();
	dwt_rxenable(DWT_START_RX_IMMEDIATE);
}

int8_t checkTO(uint64_t * last_time, uint64_t timeout){
		/* check ranging timeout and restart ranging if necessary  */
		if ((get_time_us() - *last_time) > timeout) { 
			//rx_timestamp_poll = 0; // maybe delete all this
			//tx_timestamp_poll = 0;
			//rx_timestamp_response = 0;
			//tx_timestamp_response = 0;
			//rx_timestamp_final = 0;
			//tx_timestamp_final = 0;
			tx_done = 0;
			rx_done = 0;
			return 1;
		} else {
			return 0;
		}
}


void transmit_rx_diagnostics(uint64_t ts, uint16_t id, float current_rotation, int16_t pdoa_rx, int16_t pdoa_tx, uint8_t * tdoa) {
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

	csv_write_rx2(ts, id, pdoa_read_rx, pdoa_read_tx, tdoa_read, current_rotation);
}

void print_hex(const uint8_t *bytes, size_t length) {
	for (size_t i = 0; i < length; i++) {
		printf("%02X ", bytes[i]);
	}
	printf("\n");
}

uint64_t get_time_us(){
	// returns the current time in us since the system clock started
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec*1000000UL + (uint64_t)ts.tv_nsec/1000;
}

#endif
