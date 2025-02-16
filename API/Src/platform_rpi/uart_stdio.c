/**
* @file       port_stdio.c
*
* @brief      HW specific functions for standard IO interface
*
* @attention  Copyright 2018-2019 (c) Decawave Ltd, Dublin, Ireland.
*             All rights reserved.
*
* @author     Decawave
*
* This file contains target specific implementations of functions used by the
* production test program for reading from the standard input and writing to the
* standard output. This standard I/O can be a UART peripheral, Segger RTT,
* semihosting, an LCD, ... As long a it can handle sending a data .
*/

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "uart_stdio.h"

/* Platform specific includes */
#include "main.h"

//static UART_HandleTypeDef* uart = NULL;
FILE  *logfile; 

/*! ----------------------------------------------------------------------------
 * @fn port_stdio_init
 * @brief Initialize stdio on the given UART
 *
 * @param[in] huart Pointer to the STM32 HAL UART peripheral instance
 */
void stdio_init() {
    logfile = fopen("log.csv", "w");
    assert(logfile != NULL);
}

/*! ----------------------------------------------------------------------------
 * @fn stdio_write
 * @brief Transmit/write data to standard output
 *
 * @param[in] data Pointer to null terminated string
 * @return Number of bytes transmitted or -1 if an error occurred
 */
inline int stdio_write(const char *data)
{    
    uint16_t len = strlen(data);
    fprintf(logfile, data);
    fflush(logfile);
    printf("%s", data); // also print to console
    return len;

    /*
    if (HAL_UART_Transmit(uart, (uint8_t*) data, len, HAL_MAX_DELAY) == HAL_OK) {
        return len;
    }
    return -1;
    */
}

inline int stdio_write_binary(const uint8_t *data, uint16_t length)
{
    uint8_t mask;
    for (int i=0; i<length; i++){
        mask = 128;
        for (int j=0; j<8; j++){
            if ((mask & data[i]) > 0)
            {
                fputc('1', logfile);
            } 
            else  
            {
                fputc('0', logfile);
            }
            mask = mask >> 1;
        }
    }
    fputc('\n', logfile);
    fflush(logfile);
    return length;
    /*
    if (HAL_UART_Transmit(uart, data, length, HAL_MAX_DELAY) == HAL_OK) {
        return length;
    }
    return -1;
    */
}

inline int stdio_write_hex(const uint8_t *data, uint16_t length, uint16_t offset)
{
    for (int i=offset; i<length; i++){
        //print as hexadecimal
        fprintf(logfile, "%02X", data[i]);
    }
    fputc('\n', logfile);
    fflush(logfile);
    return length;
}

inline void csv_write_rx(float pdoa, int64_t tdoa, uint16_t current_rotation){
    // type 0: rx data for pdoa/tdoa:
    fprintf(logfile, "0,%f,%ld,%d", pdoa, tdoa, current_rotation);
    fputc('\n', logfile);
    fflush(logfile);
}

inline void csv_write_twr(uint64_t Treply1, uint64_t Treply2, uint64_t Tround1, uint64_t Tround2, uint32_t dist_mm, uint16_t twr_count, uint16_t current_rotation){
    // type 1: twr data
    fprintf(logfile, "1,%lu,%lu,%lu,%lu,%u,%u,%u", Treply1, Treply2, Tround1, Tround2, dist_mm, twr_count, current_rotation);
    fputc('\n', logfile);
    fflush(logfile);
}

inline void csv_write_CIR(const uint8_t *data, uint16_t length, uint16_t offset){
    // type 2: CIR data
    fprintf(logfile, "2");
    for (int i=offset; i<length; i++){
        //print as hexadecimal
        fprintf(logfile, ",%02X", data[i]);
    }
    fputc('\n', logfile);
    fflush(logfile);
}