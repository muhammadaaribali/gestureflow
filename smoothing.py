"""Turn flickery per-frame predictions into stable, one-shot gesture events."""
from collections import Counter, deque


class GestureStabilizer:
    def __init__(self, window=7, min_votes=5, min_conf=0.7, hold_seconds=0.4,
                 release_seconds=0.4, idle_labels=("Idle",)):
        self.votes = deque(maxlen=window)
        self.min_votes = min_votes
        self.min_conf = min_conf
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds      # a held gesture survives a dropout this long
        self.idle_labels = idle_labels              # labels that mean "no gesture"
        self.candidate = None
        self.candidate_since = 0.0
        self.active = None
        self.off_since = None                       # when the active gesture last lost the vote

    def update(self, label, confidence, now):
        """Returns (active, triggered): the confirmed gesture (or None), and True
        only on the single frame where a gesture becomes confirmed."""
        # 1. Idle, low confidence or no hand all count as a "no gesture" vote
        counts = (label is not None and label not in self.idle_labels
                  and confidence >= self.min_conf)
        self.votes.append(label if counts else None)

        # 2. Majority vote with a minimum number of votes
        winner, count = Counter(self.votes).most_common(1)[0]
        stable = winner if (winner is not None and count >= self.min_votes) else None

        # 3. How long has the same stable gesture been winning?
        if stable != self.candidate:
            self.candidate = stable
            self.candidate_since = now
        held = stable is not None and now - self.candidate_since >= self.hold_seconds

        # 4. Decide what is active
        if self.active is not None:
            if stable == self.active:
                self.off_since = None               # still holding it
                return self.active, False
            if self.off_since is None:
                self.off_since = now
            if stable is None:
                # brief dropout: keep the gesture until the release time has passed
                if now - self.off_since >= self.release_seconds:
                    self.active, self.off_since = None, None
                return self.active, False
            if held and counts and label == stable:                                # another gesture has won long enough
                self.active, self.off_since = stable, None
                return self.active, True
            return self.active, False
        if held and counts and label == stable:
            self.active, self.off_since = stable, None
            return self.active, True
        return None, False
