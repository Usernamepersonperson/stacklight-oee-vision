"""Optional MQTT publishing (needs paho-mqtt). Topic: factory/<line>/<machine>/state."""

from __future__ import annotations

import json

from .state import StateEvent


class MqttPublisher:
    def __init__(self, host: str, line: str, port: int = 1883) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "MQTT output needs paho-mqtt. Install it with: pip install 'stacklight-oee-vision[mqtt]'"
            ) from exc
        self.line = line
        try:
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:  # paho-mqtt 1.x
            self._client = mqtt.Client()
        self._client.connect(host, port)
        self._client.loop_start()

    def topic(self, machine: str) -> str:
        return f"factory/{self.line}/{machine}/state"

    def publish_event(self, event: StateEvent) -> None:
        payload = json.dumps({"line": self.line, **event.to_dict()})
        self._client.publish(self.topic(event.machine), payload, qos=1, retain=True)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
