"""Dispatch confirmed gestures to a configurable backend."""
ONE_SHOT = {"play_pause", "mute", "next_track", "prev_track"}
REPEATING = {"volume_up", "volume_down"}
EPS = 1e-9


def dry_run_backend(action):
    print(f"[DRY RUN] {action}", flush=True)


class ActionController:
    def __init__(self, mapping, backend, cooldown=1.5, repeat_interval=0.3):
        for action in mapping.values():
            if action is not None and action not in ONE_SHOT | REPEATING:
                raise ValueError(f"Unknown action in config: {action!r}")
        self.mapping = mapping
        self.backend = backend
        self.cooldown = cooldown
        self.repeat_interval = repeat_interval
        self.enabled = False
        self.last_fired = {}
        self.log = []

    def update(self, active, triggered, now):
        action = self.mapping.get(active)
        if not self.enabled or active is None or action is None:
            return None
        last = self.last_fired.get(action)
        if action in REPEATING:
            due = triggered or (last is not None and now - last >= self.repeat_interval - EPS)
        else:
            due = triggered and (last is None or now - last >= self.cooldown - EPS)
        if due:
            self.backend(action)
            self.last_fired[action] = now
            self.log.append((now, active, action))
            return action
        return None
