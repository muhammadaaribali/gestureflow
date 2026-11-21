"""Confirm gestures using recent frame votes and a timed hold."""
from collections import Counter, deque


class GestureStabilizer:
    def __init__(self, window=7, min_votes=5, min_conf=0.7, hold_seconds=0.4):
        self.votes = deque(maxlen=window)
        self.min_votes = min_votes
        self.min_conf = min_conf
        self.hold_seconds = hold_seconds
        self.candidate = None
        self.candidate_since = 0.0
        self.active = None

    def update(self, label, confidence, now):
        """Return the active label and whether this frame first confirmed it."""
        vote = label if confidence >= self.min_conf else None
        self.votes.append(vote)
        winner, count = Counter(self.votes).most_common(1)[0]
        candidate = winner if count >= self.min_votes else None
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = now
            self.active = None

        # Only confirm on a frame that actually sees the winning gesture.
        if (candidate is not None and vote == candidate and self.active is None
                and now - self.candidate_since >= self.hold_seconds):
            self.active = candidate
            return self.active, True
        return self.active, False
