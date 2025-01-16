/*! ----------------------------------------------------------------------------
 * @file    deca_sleep.c
 * @brief   platform dependent sleep implementation
 *
 * @attention
 *
 * Copyright 2015-2020 (c) DecaWave Ltd, Dublin, Ireland.
 *
 * All rights reserved.
 *
 * @author DecaWave
 */

#include <deca_device_api.h>
#include <port.h>
#include <unistd.h>
#include <wiringPi.h>

/* Wrapper function to be used by decadriver. Declared in deca_device_api.h */
void deca_sleep(unsigned int time_ms)
{
    /*
     * Pause execution for 'time_ms' milliseconds. Due to the multi-tasking nature of Linux,
     * it could be longer. Maximum delay is an unsigned 32-bit integer (or ~49 days).
     */
    delay(time_ms);
}

/* Wrapper function to be used by decadriver. Declared in deca_device_api.h */
void deca_usleep(unsigned long time_us)
{
    usleep(time_us);
}
