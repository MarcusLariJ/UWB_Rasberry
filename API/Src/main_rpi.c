/*! ----------------------------------------------------------------------------
 * @file    main_rpi.c
 * @brief   Raspberry Pi build main entry point for simple examples.
 *
 * @attention
 *
 * Copyright 2020 (c) DecaWave Ltd, Dublin, Ireland.
 *
 * All rights reserved.
 *
 */

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "port.h"
#include "examples_defines.h"
#include <wiringPi.h>

#define UNIT_TEST 0

extern example_ptr example_pointer;
extern int unit_test_main(void);
extern void build_examples(void);

static void RPi_GPIO_Init(void);
static void RPi_SPI1_Init(void);

/*! ------------------------------------------------------------------------------------------------------------------
* @fn test_run_info()
*
* @brief  This function is simply a printf() call for a string. It is implemented differently on other platforms,
*         but on the RPi, the application is called from the user level and printf() is accessible.
*
* @param data - Message data, this data should be NULL string.
*
* output parameters
*
* no return value
*/
void test_run_info(unsigned char *data)
{
    printf("%s\n", data);
}

/**
  * @brief  The application entry point.
  *
  * @retval None
  */
int main(void)
{
    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */

    /* MCU Configuration----------------------------------------------------------*/

    /* Reset of all peripherals (if attached). */
    build_examples();

    /* USER CODE BEGIN Init */

    /* USER CODE END Init */

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    RPi_GPIO_Init();
    RPi_SPI1_Init();

    if(UNIT_TEST)
    {
        unit_test_main();
    }
    else
    {
        //Run the selected example as selected in example_selection.h
        example_pointer();
    }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
    while (1)
    {

  /* USER CODE END WHILE */

  /* USER CODE BEGIN 3 */

    }
  /* USER CODE END 3 */

}

/**
 * @brief  Initialise the Raspberry Pi's GPIO interface using WiringPi.
*/
static void RPi_GPIO_Init(void)
{
    /* Setup the WiringPi interface */
    wiringPiSetup();
    if (wiringPiSetup () < 0)
    {
        fprintf(stderr, "Unable to set up wiringPi: %s\n", strerror(errno));
    }

    /*
     * WiringPi SPI setup will handle the Chip Enable for the SPI comms.
     * No need to enable a GPIO port for it here.
     */

    /* Configure Reset GPIO as input */
    pinMode(DW_RESET_Pin, INPUT);
    pullUpDnControl(DW_RESET_Pin, PUD_OFF);

    /* Configure Wakeup GPIO as output */
    pinMode(DW_WAKEUP_Pin, OUTPUT);
    pullUpDnControl(DW_WAKEUP_Pin, PUD_OFF);
    /*
     * Set up the interrupt with a simple counter function for now.
     * We can replace this function with another if required.
     */
    pinMode(DW_IRQn_Pin, INPUT);
    pullUpDnControl(DW_IRQn_Pin, PUD_DOWN);
}

/**
 * @brief  Initialise the Raspberry Pi's SPI interface using WiringPi.
*/
static void RPi_SPI1_Init(void)
{
    int fd = wiringPiSPISetupMode(SPI_CHANNEL, SPI_CLOCK_SPEED_FAST, 0);
    if (fd == -1) {
        fprintf(stderr, "Failed to init SPI communication: %s\n", strerror(errno));
    }
    printf("SPI communication successfully setup.\n");
}
