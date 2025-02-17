#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>
#include <stdint.h>

#define SPI_CHANNEL 1
#define SPI_CLOCK_SPEED_SLOW 2000000
#define CMD_LEN 2
#define BIT2DEG 0.0219739974364

uint16_t read_encoder(uint16_t cmd){
    // 16 bits -> 2 bytes
    uint8_t chk;
    uint8_t buffer[2];
    // commands are read with MSB first - thus switch the to bytes around
    buffer[0] = (uint8_t)(cmd >> 8);
    buffer[1] = (uint8_t)cmd;
    chk = wiringPiSPIDataRW(SPI_CHANNEL, buffer, 2);
    // The contents of the buffer is now replaced with the response
    uint16_t contents = (uint16_t)buffer[1] + ((uint16_t)buffer[0] << 8);
    return contents;
} 

int main()
{
    printf("Start\n");
    
    // setup
    uint8_t chk;
    chk = wiringPiSetup();
    if (chk == -1) {return(-1);}
    chk = wiringPiSPISetupMode(SPI_CHANNEL, SPI_CLOCK_SPEED_SLOW, 1);
    if (chk == -1) {return(-1);}
    delay(20);
    uint16_t output;
    double degrees;
    uint8_t alarmbits;

    // commands:
    uint16_t cmd_NOP = 0;
    uint16_t cmd_getAngle = (0x3fff | 0x4000) | 0x8000; // 1 parity (for even), 1 for read, read angle at addresss 0x3fff
    uint16_t cmd_clear = (0x4001); // command for clearing error bit

    // clear errors
    output = read_encoder(cmd_clear); // this output is going to be trash anyways

    while (1){
        // keep reading angles
        output = read_encoder(cmd_getAngle);
        delay(20);
        output = read_encoder(cmd_NOP); // extra NOP to get current value. Maybe increase buffer instead
        alarmbits = (uint8_t)((output >> 14) & 3);
        if (alarmbits & 1) 
        {
            printf("alarm bits on\n");
            printf("%d\n", alarmbits);
        }
        // Print response value
        // Convert to angle:
        degrees = (double)(output & 0x3fff) * (360.0 / 0x4000);
        printf("Angle: \n");
        printf("%.2f\n", degrees);
        delay(200);
    }
    
    
    
    
    /*
    // Write to the encoder:
    uint8_t readBuffer[CMD_LEN];
    uint8_t headerBuffer[CMD_LEN];
    uint8_t buf[CMD_LEN + CMD_LEN]; // the buffer buf is used both for transmission and recieving over the SPI

    // Clear error flags
    memcpy(headerBuffer, &cmd_clear, CMD_LEN);
    memcpy(buf, headerBuffer, CMD_LEN);
    wiringPiSPIDataRW(SPI_CHANNEL, buf, (CMD_LEN+CMD_LEN));
    memcpy(readBuffer, &buf[CMD_LEN], CMD_LEN);

    // Keep sampling for angles:
    memcpy(headerBuffer, &cmd_getAngle, CMD_LEN);
    while(1)
    {
        memcpy(buf, headerBuffer, CMD_LEN);
        wiringPiSPIDataRW(SPI_CHANNEL, buf, (CMD_LEN + CMD_LEN));
        memcpy(readBuffer, &buf[CMD_LEN], CMD_LEN);

        // Combine bytes to read 14 bit number. Remove two upper alarm bits
        uint8_t alarmbits = (readBuffer[1] >> 6) & 3;
        if (alarmbits & 1){
            printf("alarm bits on\n");
            printf("%d\n", alarmbits);
        }
        uint16_t angle = readBuffer[0] + (readBuffer[1] & 0x3f) << 8;
        printf("Angle: \n");
        printf("%d\n", angle);
        delay(1000);
    }
    */
}