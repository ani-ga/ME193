// MCU (STM32) side of the Activity Tracker app.
//
// Job: read the MPU6050 over I2C as fast as we reasonably can, and hand the
// latest reading to Python whenever it asks. All the real work (windowing,
// features, KNN, the web page) happens in python/main.py on the Linux side,
// reached over the Bridge.
//
// Six raw values are exposed, one accelerometer/gyro axis each, so each
// Bridge-provided function returns a single plain int - the simplest type
// the Bridge is known to carry reliably. Python does the unit conversion
// (raw register counts -> g's and deg/s) since that's easier to read there.

#include <Wire.h>
#include "Arduino_RouterBridge.h"

const uint8_t MPU_ADDR = 0x68;        // MPU6050's I2C address (default, AD0 pin low)
const uint8_t REG_PWR_MGMT_1 = 0x6B;  // power management register
const uint8_t REG_ACCEL_XOUT_H = 0x3B;  // first of 14 consecutive data registers

// Latest readings, refreshed continuously in loop(). Bridge calls just read
// these - they never block on I2C, so Python always gets a fast answer.
volatile int16_t ax = 0, ay = 0, az = 0;
volatile int16_t gx = 0, gy = 0, gz = 0;

int read_ax() { return ax; }
int read_ay() { return ay; }
int read_az() { return az; }
int read_gx() { return gx; }
int read_gy() { return gy; }
int read_gz() { return gz; }

void setup() {
  Wire.begin();

  // Wake the MPU6050 up - it starts in sleep mode.
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0);
  Wire.endTransmission(true);

  Bridge.begin();
  Bridge.provide("read_ax", read_ax);
  Bridge.provide("read_ay", read_ay);
  Bridge.provide("read_az", read_az);
  Bridge.provide("read_gx", read_gx);
  Bridge.provide("read_gy", read_gy);
  Bridge.provide("read_gz", read_gz);
}

void loop() {
  // Ask for the 14 bytes starting at ACCEL_XOUT_H: accel (6 bytes),
  // temperature (2 bytes, unused here), gyro (6 bytes).
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 14, true);

  if (Wire.available() >= 14) {
    ax = (int16_t)((Wire.read() << 8) | Wire.read());
    ay = (int16_t)((Wire.read() << 8) | Wire.read());
    az = (int16_t)((Wire.read() << 8) | Wire.read());
    Wire.read(); Wire.read();  // skip the temperature reading
    gx = (int16_t)((Wire.read() << 8) | Wire.read());
    gy = (int16_t)((Wire.read() << 8) | Wire.read());
    gz = (int16_t)((Wire.read() << 8) | Wire.read());
  }

  delay(5);  // ~200 Hz refresh; Python polls slower than this, so it always sees a fresh value
}
