// Standalone servo test (no Bridge, no Python). Sweeps 0 -> 180 -> 0 slowly.
// Servo signal on D9. Servo power comes from its OWN 5 V pack, with GND shared to the UNO Q.
#include <Servo.h>

// ---- config (keep in sync with config.py) ----
const int PIN_SERVO = 9;
const int PULSE_MIN_US = 500;    // MG995 range; if it doesn't reach the ends, widen these
const int PULSE_MAX_US = 2500;
const int LOCKED_ANGLE = 0;
const int OPEN_ANGLE = 180;

Servo servo;

void setup() {
  servo.attach(PIN_SERVO, PULSE_MIN_US, PULSE_MAX_US);
  servo.write(LOCKED_ANGLE);
  delay(1000);
}

void loop() {
  for (int a = LOCKED_ANGLE; a <= OPEN_ANGLE; a += 5) { servo.write(a); delay(40); }
  delay(1000);
  for (int a = OPEN_ANGLE; a >= LOCKED_ANGLE; a -= 5) { servo.write(a); delay(40); }
  delay(1000);
}
