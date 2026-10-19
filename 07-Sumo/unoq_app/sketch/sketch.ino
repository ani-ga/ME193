// UnoQ side (microcontroller half): nothing to do. The LEGO Color Sensor
// connects to the Linux half over Bluetooth, and python/main.py reads it and
// publishes it over MQTT. This sketch only starts the Bridge so App Lab can
// run the app.

#include <Arduino_RouterBridge.h>

void setup() {
  Bridge.begin();
}

void loop() {
  delay(100);
}
