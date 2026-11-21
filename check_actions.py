from actions import ActionController


def make(enabled=True):
    fired = []
    controller = ActionController(
        {"Open": "play_pause", "Pointer": "volume_up", "Close": None}, fired.append
    )
    controller.enabled = enabled
    return controller, fired


def check(name, ok):
    if not ok:
        raise AssertionError("FAIL " + name)
    print("PASS " + name)


c, fired = make(enabled=False)
c.update("Open", True, 0.0)
check("nothing fires while control is OFF", fired == [])

c, fired = make()
c.update("Open", True, 0.0)
for i in range(1, 21):
    c.update("Open", False, i * 0.1)
check("one-shot fires once while the gesture is held", fired == ["play_pause"])

c, fired = make()
for t in (0.0, 1.0, 1.6):
    c.update("Open", True, t)
check("one-shot cooldown blocks the 1.0 s repeat, allows 1.6 s", fired == ["play_pause"] * 2)

c, fired = make()
c.update("Pointer", True, 0.0)
for i in range(1, 11):
    c.update("Pointer", False, i * 0.1)
check("volume repeats every 0.3 s while held (4 times in 1 s)", fired == ["volume_up"] * 4)

c, fired = make()
c.update("Pointer", True, 0.0)
c.update(None, False, 0.5)
c.update(None, False, 1.0)
check("nothing fires after the gesture is released", fired == ["volume_up"])

c, fired = make()
c.update("Close", True, 0.0)
check("a gesture mapped to null does nothing", fired == [])
