import time

from mqttlib import MQTTClient

TOPIC = "ME193"

def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")


with MQTTClient() as client:
    client.subscribe(TOPIC, on_message, qos=1)
    time.sleep(1)  # give the subscription time to reach the broker

    client.publish(TOPIC, "2", qos=1)
    print(f"Published '2' to '{TOPIC}' on test.mosquitto.org")

    print(f"Listening on '{TOPIC}'... press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
