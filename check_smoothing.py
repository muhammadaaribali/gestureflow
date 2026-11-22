"""Replay gesture votes at 30 frames per second without a camera."""
from smoothing import GestureStabilizer


def check(name, frames, expected_events, expected_active):
    stabilizer = GestureStabilizer()
    events = []
    for frame, (label, confidence) in enumerate(frames):
        active, triggered = stabilizer.update(label, confidence, frame / 30)
        if triggered:
            events.append(active)
    assert (events, active) == (expected_events, expected_active), (name, events, active)
    print("PASS", name)


if __name__ == "__main__":
    palm = [("Open", 0.95)]
    fist = [("Close", 0.95)]
    missing = [(None, 0.0)]
    check("held gesture fires once", palm * 45, ["Open"], "Open")
    check("single-frame glitch is ignored", palm * 15 + fist + palm * 15, ["Open"], "Open")
    check("low confidence never fires", [("Ok", 0.5)] * 30, [], None)
    check("hand loss clears the gesture", palm * 20 + missing * 20, ["Open"], None)
    check("switching gesture fires once each", palm * 20 + fist * 30, ["Open", "Close"], "Close")
    check("hand removed before confirmation", palm * 16 + missing * 10, [], None)
    check("a 0.2 s detection gap does not re-trigger a held gesture",
          palm * 20 + missing * 6 + palm * 20, ["Open"], "Open")
    check("a 1 s gap releases the gesture and it can fire again",
          palm * 20 + missing * 30 + palm * 20, ["Open", "Open"], "Open")
    idle = [("Idle", 0.99)]
    check("Idle predictions never fire", idle * 45, [], None)
    check("switching to Idle releases the gesture", palm * 20 + idle * 20, ["Open"], None)
