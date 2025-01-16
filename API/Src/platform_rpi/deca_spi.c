/*! ----------------------------------------------------------------------------
 * @file    deca_spi.c
 * @brief   SPI access functions
 *
 * @attention
 *
 * Copyright 2015-2020 (c) DecaWave Ltd, Dublin, Ireland.
 *
 * All rights reserved.
 *
 * @author DecaWave
 */

#include <deca_spi.h>
#include <deca_device_api.h>
#include <port.h>
#include <wiringPiSPI.h>

#define SPI_CHANNEL 0
#define SPI_CLOCK_SPEED 32000000
#define CRC_SIZE 1


/****************************************************************************//**
 *
 *                              DW1000 SPI section
 *
 *******************************************************************************/
/*! ------------------------------------------------------------------------------------------------------------------
 * Function: openspi()
 *
 * Low level abstract function to open and initialise access to the SPI device.
 * returns 0 for success, or -1 for error
 */
int openspi(/*SPI_TypeDef* SPIx*/)
{
    return 0;
} // end openspi()

/*! ------------------------------------------------------------------------------------------------------------------
 * Function: closespi()
 *
 * Low level abstract function to close the the SPI device.
 * returns 0 for success, or -1 for error
 */
int closespi(void)
{
    return 0;
} // end closespi()




/*! ------------------------------------------------------------------------------------------------------------------
 * Function: writetospiwithcrc()
 *
 * Low level abstract function to write to the SPI when SPI CRC mode is used
 * Takes two separate byte buffers for write header and write data, and a CRC8 byte which is written last
 * returns 0 for success, or -1 for error
 */
int writetospiwithcrc(
                uint16_t      headerLength,
                const uint8_t *headerBuffer,
                uint16_t      bodyLength,
                const uint8_t *bodyBuffer,
                uint8_t       crc8)
{
    decaIrqStatus_t  stat ;
    stat = decamutexon() ;

    uint8_t buf[headerLength + bodyLength + CRC_SIZE];
    memcpy(buf, headerBuffer, headerLength);

    if(bodyLength != 0)
    {
        memcpy(&buf[headerLength], bodyBuffer, bodyLength);
        headerLength += bodyLength;
    }
    buf[headerLength] = crc8;
    wiringPiSPIDataRW(SPI_CHANNEL, buf, (headerLength + CRC_SIZE));

    decamutexoff(stat);
    return 0;
} // end writetospiwithcrc()


/*! ------------------------------------------------------------------------------------------------------------------
 * Function: writetospi()
 *
 * Low level abstract function to write to the SPI
 * Takes two separate byte buffers for write header and write data
 * returns 0 for success, or -1 for error
 */
int writetospi(uint16_t       headerLength,
               const uint8_t  *headerBuffer,
               uint16_t       bodyLength,
               const uint8_t  *bodyBuffer)
{
    decaIrqStatus_t  stat ;
    stat = decamutexon() ;

    uint8_t buf[headerLength + bodyLength];
    memcpy(buf, headerBuffer, headerLength);
    if(bodyLength != 0)
    {
        memcpy(&buf[headerLength], bodyBuffer, bodyLength);
        headerLength += bodyLength;
    }
    wiringPiSPIDataRW(SPI_CHANNEL, buf, headerLength);

    decamutexoff(stat);
    return 0;
} // end writetospi()


/*! ------------------------------------------------------------------------------------------------------------------
 * Function: readfromspi()
 *
 * Low level abstract function to read from the SPI
 * Takes two separate byte buffers for write header and read data
 * returns the offset into read buffer where first byte of read data may be found,
 * or returns -1 if there was an error
 */
//#pragma GCC optimize ("O3")
int readfromspi(uint16_t headerLength,
                uint8_t  *headerBuffer,
                uint16_t readlength,
                uint8_t  *readBuffer)
{
    decaIrqStatus_t  stat ;
    stat = decamutexon() ;

    uint8_t buf[headerLength + readlength];

    /* Send header */
    memcpy(buf, headerBuffer, headerLength);
    wiringPiSPIDataRW(SPI_CHANNEL, buf, (headerLength + readlength));
    memcpy(readBuffer, &buf[headerLength], readlength);

    decamutexoff(stat);

    return 0;
} // end readfromspi()

/****************************************************************************//**
 *
 *                              END OF DW1000 SPI section
 *
 *******************************************************************************/


