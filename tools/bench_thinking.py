"""Discover explicit thinking-off support once and label the requested benchmark mode."""
import json
import urllib.request

from bench_auth import authorize


def discover(base: str, model: str, api_key: str | None = None) -> bool | None:
    """Only this model's explicit boolean is trusted; old or unavailable discovery remains unknown."""
    try:
        request = urllib.request.Request(base.rstrip("/") + "/v1/models", headers={"Accept": "application/json"})
        authorize(request, api_key)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            return None
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            return None
        bits = [entry.get("capabilities", {}).get("thinking_switch")
                for entry in document["data"] if isinstance(entry, dict) and entry.get("id") == model
                and isinstance(entry.get("capabilities"), dict)]
        if bits and all(type(bit) is bool and bit is bits[0] for bit in bits):
            return bits[0]
    except (OSError, ValueError, TypeError):
        pass
    return None


def configure(items: list[dict], switch: bool | None) -> list[dict]:
    """Keep the shared prompt definitions unchanged across repeated runs."""
    return [{**item, "thinking_switch": switch} for item in items]


def mode(item: dict) -> str:
    """A default is never described as off when the server did not advertise that capability."""
    if item["kind"] != "chat":
        return "not applicable"
    switch = item.get("thinking_switch")
    if switch is True:
        return "off"
    return "default (thinking off unsupported)" if switch is False else "default (thinking switch unknown)"


def receipt(item: dict) -> dict:
    return {"thinking": mode(item), "thinking_off_requested": item["kind"] == "chat" and item.get("thinking_switch") is True,
            "thinking_switch": item.get("thinking_switch")}
