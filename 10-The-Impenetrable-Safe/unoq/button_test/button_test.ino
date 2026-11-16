// Button test. Button wired between D2 and GND (no resistor needed: INPUT_PULLUP).
// Pressing the button toggles the board's built-in LED and prints to Serial.
const int PIN_BUTTON = 2;

bool lastPressed = false;
bool ledOn = false;

void setup() {
  pinMode(PIN_BUTTON, INPUT_PULLUP);   // pressed = LOW
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  bool pressed = (digitalRead(PIN_BUTTON) == LOW);
  if (pressed && !lastPressed) {       // act only on the moment of pressing
    ledOn = !ledOn;
    digitalWrite(LED_BUILTIN, ledOn ? HIGH : LOW);
    Serial.println("button pressed");
  }
  lastPressed = pressed;
  delay(20);                           // simple debounce
}
