"""Publish a voice announcement over MQTT and wait for v2v's ack — the same round trip
a Home Assistant automation does (mqtt.publish -> wait_for_trigger on the ack topic).

Usage:  .venv\\Scripts\\python.exe tools\\mqtt_announce_test.py "Zachary is home." [character] [cooldownKey]
Reads broker/topic from voice_devices.json (voice_mqtt).
"""
import json
import os
import sys
import time

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(HERE, "voice_devices.json"), encoding="utf-8"))["voice_mqtt"]

text = sys.argv[1] if len(sys.argv) > 1 else "M Q T T announcer test."
character = sys.argv[2] if len(sys.argv) > 2 else "jerma"
cooldown_key = sys.argv[3] if len(sys.argv) > 3 else ""
reply_id = f"test-{int(time.time() * 1000)}"
ack_topic = f"{cfg['ack_topic']}/{reply_id}"
body = {"device": "echo-dot-biscuit", "text": text, "character": character,
        "name": "MQTT test", "replyId": reply_id}
if cooldown_key:
    body["cooldownKey"] = cooldown_key
    body["cooldownSeconds"] = 600

got = {}


def on_connect(c, u, f, rc, p=None):
    c.subscribe(ack_topic, qos=1)
    c.publish(cfg["topic"], json.dumps(body), qos=1)
    print(f"published to {cfg['topic']} replyId={reply_id}; waiting on {ack_topic}")


def on_message(c, u, m):
    got["ack"] = json.loads(m.payload.decode())
    c.disconnect()


c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.on_connect = on_connect
c.on_message = on_message
c.connect(cfg["broker"], int(cfg.get("port", 1883)), keepalive=30)
t0 = time.time()
c.loop_start()
while "ack" not in got and time.time() - t0 < 60:
    time.sleep(0.2)
c.loop_stop()
if "ack" not in got:
    print("NO ACK within 60s")
    sys.exit(1)
a = got["ack"]
print(f"ack after {time.time() - t0:.1f}s: status={a.get('status')} delivered={a.get('delivered')} "
      f"error={a.get('error')} played_ms={a.get('played_ms')} voice={a.get('voice')} "
      f"timing={a.get('timing')}")
