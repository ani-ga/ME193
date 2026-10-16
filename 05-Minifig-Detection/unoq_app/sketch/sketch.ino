// UnoQ side (microcontroller half): drives the two DC motors and draws the
// minifig's position as a dot on the 13x8 LED matrix. The Python half
// (python/main.py) decides what to do and calls these over the Bridge:
//
//   drive(speed)  speed -100..100, same on both wheels (+ = forward)
//   dot(col, row) light one LED, or clear the matrix when col < 0
//
// Wiring: Cytron dual motor driver.
//
//   MDD3A / Maker Drive (inputs M1A, M1B, M2A, M2B) -- the default:
//     M1A -> D3    M1B -> D5     (left motor on M1A/M1B outputs)
//     M2A -> D6    M2B -> D9     (right motor on M2A/M2B outputs)
//   MDD10A (inputs PWM1, DIR1, PWM2, DIR2) -- set CYTRON_MDD10A to true:
//     PWM1 -> D3   DIR1 -> D4    (left motor on M1A/M1B outputs)
//     PWM2 -> D6   DIR2 -> D7    (right motor on M2A/M2B outputs)
//
//   Driver GND -> UnoQ GND (they must share ground).
//   Motor battery + / - -> driver VB+ / VB- (never power motors from the UnoQ).

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const bool CYTRON_MDD10A = false;  // false = MDD3A / Maker Drive
// For MDD3A these are the A/B inputs (both PWM); for MDD10A, PWM and DIR.
const int LEFT_A = 3, LEFT_B = CYTRON_MDD10A ? 4 : 5;
const int RIGHT_A = 6, RIGHT_B = CYTRON_MDD10A ? 7 : 9;
// Set to true for a motor that spins backward when it should go forward.
const bool LEFT_REVERSED = true;
const bool RIGHT_REVERSED = true;

const unsigned long WATCHDOG_MS = 1000;  // stop if Python goes quiet

const int MATRIX_COLS = 13, MATRIX_ROWS = 8;
const uint8_t DOT_BRIGHTNESS = 7;

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_COLS * MATRIX_ROWS];
volatile unsigned long lastDriveMs = 0;

void setMotor(int pinA, int pinB, bool reversed, int speed) {
  if (reversed) speed = -speed;
  int duty = map(constrain(abs(speed), 0, 100), 0, 100, 0, 255);
  if (CYTRON_MDD10A) {
    // pinA = PWM (speed), pinB = DIR (direction).
    digitalWrite(pinB, speed < 0 ? HIGH : LOW);
    analogWrite(pinA, duty);
  } else {
    // PWM on A spins one way, PWM on B the other; both low = brake.
    analogWrite(pinA, speed > 0 ? duty : 0);
    analogWrite(pinB, speed < 0 ? duty : 0);
  }
}

void drive(int speed) {
  setMotor(LEFT_A, LEFT_B, LEFT_REVERSED, speed);
  setMotor(RIGHT_A, RIGHT_B, RIGHT_REVERSED, speed);
  lastDriveMs = millis();
}

void dot(int col, int row) {
  memset(frame, 0, sizeof(frame));
  if (col >= 0 && col < MATRIX_COLS && row >= 0 && row < MATRIX_ROWS) {
    frame[row * MATRIX_COLS + col] = DOT_BRIGHTNESS;
  }
  matrix.draw(frame);
}

void setup() {
  int pins[] = {LEFT_A, LEFT_B, RIGHT_A, RIGHT_B};
  for (int pin : pins) pinMode(pin, OUTPUT);
  drive(0);

  matrix.begin();
  dot(-1, -1);

  Bridge.begin();
  Bridge.provide("drive", drive);
  Bridge.provide("dot", dot);
}

void loop() {
  if (millis() - lastDriveMs > WATCHDOG_MS) {
    drive(0);
  }
  delay(50);
}
