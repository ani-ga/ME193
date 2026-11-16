// MCU (STM32) side of the safe. It does the physical things; Python decides WHEN.
//   Python calls  open_door()   -> servo to OPEN_ANGLE
//   Python calls  close_door()  -> servo to LOCKED_ANGLE
//   Python calls  button_pressed() -> 1 if the button was pressed since the last call, else 0
//   Python calls  draw(bytes)   -> shows a picture on the 8 x 13 LED matrix (1 byte per pixel, 0-7 brightness)
#include <Servo.h>
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

// ---- config (keep in sync with config.py) ----
const int PIN_SERVO  = 9;      // servo signal (servo power comes from its own 5 V pack, GND shared)
const int PIN_BUTTON = 2;      // button between D2 and GND (internal pull-up, pressed = LOW)
const int PULSE_MIN_US = 500;  // MG995 pulse range
const int PULSE_MAX_US = 2500;
const int LOCKED_ANGLE = 0;
const int OPEN_ANGLE   = 180;

Servo servo;
Arduino_LED_Matrix matrix;
volatile bool pressedFlag = false;   // set in loop(), cleared when Python reads it
bool lastDown = false;

void open_door()  { servo.write(OPEN_ANGLE); }
void close_door() { servo.write(LOCKED_ANGLE); }

// Bridge calls run on their own thread; the matrix is only ever touched from here, so no lock is needed.
void draw(std::vector<uint8_t> frame) {
  if (frame.size() >= 104) matrix.draw(frame.data());   // 8 rows x 13 columns
}

int button_pressed() {
  int r = pressedFlag ? 1 : 0;
  pressedFlag = false;
  return r;
}

void setup() {
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  servo.attach(PIN_SERVO, PULSE_MIN_US, PULSE_MAX_US);
  servo.write(LOCKED_ANGLE);                 // always start locked

  matrix.begin();
  matrix.setGrayscaleBits(3);                // brightness values 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("open_door", open_door);
  Bridge.provide("close_door", close_door);
  Bridge.provide("button_pressed", button_pressed);
  Bridge.provide("draw", draw);
}

void loop() {
  bool down = (digitalRead(PIN_BUTTON) == LOW);
  if (down && !lastDown) pressedFlag = true;  // remember the moment of pressing
  lastDown = down;
  delay(10);                                  // simple debounce
}
