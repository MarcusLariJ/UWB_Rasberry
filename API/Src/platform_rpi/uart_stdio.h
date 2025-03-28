/*! ----------------------------------------------------------------------------
 * @file      stdio.h
 *
 * @brief     Functions modfied to just print to a log on the Raspberry
 *
 * @author    Decawave
 *
 * @attention Copyright 2020 (c) Decawave Ltd, Dublin, Ireland.
 *            All rights reserved.
 */

#ifndef _PORT_STDIO_H_
#define _PORT_STDIO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>

extern FILE *logfile;

/* Platform specific includes */
//#include "stm32f4xx_hal.h"

/*! ----------------------------------------------------------------------------
 * @fn stdio_init
 * @brief Initialize stdio on the given UART
 *
 * @param[in] huart Pointer to the STM32 HAL UART peripherial instance
 */
void stdio_init();

/*! ----------------------------------------------------------------------------
 * @fn stdio_write
 * @brief Transmit/write data to standard output
 *
 * @param[in] data Pointer to null terminated string
 * @return Number of bytes transmitted or -1 if an error occured
 */
int stdio_write(const char *data);

int stdio_write_binary(const uint8_t *data, uint16_t length);

int stdio_write_hex(const uint8_t *data, uint16_t length, uint16_t offset);

void csv_write_rx(float pdoa, int64_t tdoa, float current_rotation);

void csv_write_twr(uint64_t Treply1, uint64_t Treply2, uint64_t Tround1, uint64_t Tround2, uint32_t dist_mm, uint16_t twr_count, uint16_t current_rotation);

void csv_write_CIR(const uint8_t *data, uint16_t length, uint16_t offset);

#ifdef __cplusplus
}
#endif

#endif /* _PORT_STDIO_H_ */
