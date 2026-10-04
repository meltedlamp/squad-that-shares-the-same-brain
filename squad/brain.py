"""One network for the whole squad. A correction stays on this computer."""

import os
from pathlib import Path

import numpy as np

from .settings import (
    DIR_FEATURES,
    F_EDGE,
    F_HAZARD,
    F_ROUND,
    H1,
    H2,
    IN_DIM,
    OUTPUTS,
    WAIT,
)

FACTORY = Path(__file__).resolve().parent / "weights.npz"
REPLAY_MAX = 24
LEARN_LR = 0.08
REPLAY_BATCH = 6


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _user_dirs():
    found = []
    base = os.environ.get("LOCALAPPDATA")
    if base:
        found.append(Path(base) / "SquadBrain")
    found.append(Path.home() / ".squad-brain")
    return found


def _he(rng, fan_in, fan_out):
    scale = np.sqrt(2.0 / fan_in)
    return (rng.standard_normal((fan_out, fan_in)) * scale).astype(np.float32)


def rotate_action(action, turns):
    action = int(action)
    if action == WAIT:
        return WAIT
    return (action + int(turns)) % 4


def rotate_obs(obs, turns):
    """Turn the view clockwise by quarter-turns. The door turns with it."""
    turns = int(turns) % 4
    out = np.array(obs, dtype=np.float32, copy=True)
    if turns == 0:
        return out
    blocks = out[: 4 * DIR_FEATURES].reshape(4, DIR_FEATURES)
    out[: 4 * DIR_FEATURES] = np.roll(blocks, turns, axis=0).reshape(-1)
    dx = float(out[-3])
    dy = float(out[-2])
    for _ in range(turns):
        dx, dy = -dy, dx
    out[-3] = dx
    out[-2] = dy
    return out


# Pit, spikes, crack, and a bare danger. A lesson is studied as each of these.
SHAPES = (
    (1.0, 0.0),
    (0.08, 1.0),
    (0.45, 0.35),
    (0.0, 0.0),
)


def restamp(obs, round_v, edge_v):
    """Keep the danger bit, and rewrite the outline so another trap can share the step."""
    out = np.array(obs, dtype=np.float32, copy=True)
    blocks = out[: 4 * DIR_FEATURES].reshape(4, DIR_FEATURES)
    danger = blocks[:, F_HAZARD] >= 0.5
    blocks[danger, F_ROUND] = round_v
    blocks[danger, F_EDGE] = edge_v
    return out


class Brain:
    def __init__(self, weights, anchor_x, anchor_y, replay_x, replay_y):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights
        self.anchor_x = np.asarray(anchor_x, dtype=np.float32)
        self.anchor_y = np.asarray(anchor_y, dtype=np.int64)
        self.replay_x = np.asarray(replay_x, dtype=np.float32).reshape(-1, IN_DIM)
        self.replay_y = np.asarray(replay_y, dtype=np.int64).reshape(-1)

    @classmethod
    def create(cls, rng):
        weights = (
            _he(rng, IN_DIM, H1),
            np.zeros(H1, dtype=np.float32),
            _he(rng, H1, H2),
            np.zeros(H2, dtype=np.float32),
            _he(rng, H2, OUTPUTS),
            np.zeros(OUTPUTS, dtype=np.float32),
        )
        empty_x = np.zeros((0, IN_DIM), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return cls(weights, empty_x, empty_y, empty_x, empty_y)

    def copy(self):
        weights = tuple(
            part.copy() for part in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        )
        return Brain(
            weights,
            self.anchor_x.copy(),
            self.anchor_y.copy(),
            self.replay_x.copy(),
            self.replay_y.copy(),
        )

    @classmethod
    def load_factory(cls):
        factory = _read_npz(FACTORY)
        if factory is None or "anchor_x" not in factory.files or not _weights_ok(factory):
            if factory is not None:
                factory.close()
            raise FileNotFoundError(
                "The squad has no factory brain. From this folder run: python -m squad.train"
            )
        try:
            anchor_x = np.array(factory["anchor_x"], dtype=np.float32)
            anchor_y = np.array(factory["anchor_y"], dtype=np.int64)
            weights = _weights_of(factory)
        finally:
            factory.close()
        empty_x = np.zeros((0, IN_DIM), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return cls(weights, anchor_x, anchor_y, empty_x, empty_y)

    @classmethod
    def load(cls):
        factory = cls.load_factory()
        learned = None
        for folder in _user_dirs():
            path = folder / "brain.npz"
            if not path.exists():
                continue
            data = _read_npz(path)
            if data is not None and _weights_ok(data):
                learned = data
                break
            if data is not None:
                data.close()
        if learned is None:
            return factory
        try:
            weights = _weights_of(learned)
            if "replay_x" in learned.files:
                replay_x = np.array(learned["replay_x"], dtype=np.float32)
                replay_y = np.array(learned["replay_y"], dtype=np.int64)
            else:
                replay_x = np.zeros((0, IN_DIM), dtype=np.float32)
                replay_y = np.zeros((0,), dtype=np.int64)
        finally:
            learned.close()
        return cls(weights, factory.anchor_x, factory.anchor_y, replay_x, replay_y)

    @staticmethod
    def user_path():
        for folder in _user_dirs():
            path = folder / "brain.npz"
            if path.exists():
                return path
        return None

    @staticmethod
    def forget():
        for folder in _user_dirs():
            path = folder / "brain.npz"
            if path.exists():
                path.unlink()

    def save_factory(self, anchor_x, anchor_y):
        np.savez_compressed(
            FACTORY,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            anchor_x=np.asarray(anchor_x, dtype=np.float32),
            anchor_y=np.asarray(anchor_y, dtype=np.int64),
        )

    def save_user(self):
        payload = dict(
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            replay_x=self.replay_x,
            replay_y=self.replay_y,
        )
        errors = []
        for folder in _user_dirs():
            try:
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "brain.npz", **payload)
                return
            except OSError as err:
                errors.append(err)
        if errors:
            raise errors[-1]

    def choose(self, obs, legal):
        logits = self._forward(np.asarray(obs, dtype=np.float32).reshape(1, IN_DIM))[-1][0]
        masked = logits.copy()
        allowed = np.asarray(legal, dtype=bool)
        masked[~allowed] = -1.0e9
        probs = _softmax(masked.reshape(1, -1))[0]
        return int(np.argmax(probs)), probs

    def learn(self, obs, label, rng=None, save=True):
        """A few steps on this order, turned to every side, plus older orders and safe walks."""
        if rng is None:
            rng = np.random.default_rng()
        sample = np.asarray(obs, dtype=np.float32).reshape(IN_DIM)
        label = int(label)
        self.replay_x = np.concatenate([self.replay_x, sample.reshape(1, IN_DIM)], axis=0)[-REPLAY_MAX:]
        self.replay_y = np.concatenate(
            [self.replay_y, np.array([label], dtype=np.int64)]
        )[-REPLAY_MAX:]

        self._study(sample, label, LEARN_LR)
        older = len(self.replay_y) - 1
        take = min(REPLAY_BATCH, older)
        if take > 0:
            pick = rng.choice(older, size=take, replace=False)
            for row, action in zip(self.replay_x[pick], self.replay_y[pick]):
                self._study_case(row, int(action), LEARN_LR)
        if save:
            self.save_user()

    def _study(self, obs, label, lr):
        """One undiluted step for each facing, and for each trap outline."""
        sample = np.asarray(obs, dtype=np.float32).reshape(IN_DIM)
        for turns in range(4):
            turned = rotate_obs(sample, turns)
            act = rotate_action(label, turns)
            cases = [turned] + [restamp(turned, round_v, edge_v) for round_v, edge_v in SHAPES]
            for row in cases:
                self._study_case(row, act, lr)

    def _study_case(self, row, label, lr):
        row = np.asarray(row, dtype=np.float32).reshape(1, IN_DIM)
        count = len(self.anchor_y)
        if count == 0:
            self.sgd_step(np.repeat(row, 8, axis=0), np.full(8, int(label), dtype=np.int64), lr)
            return
        lesson = np.repeat(row, count, axis=0)
        labels = np.full(count, int(label), dtype=np.int64)
        self.sgd_step(
            np.concatenate([lesson, self.anchor_x]),
            np.concatenate([labels, self.anchor_y]),
            lr,
        )

    def _forward(self, batch):
        z1 = batch @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0)
        logits = h2 @ self.W3.T + self.b3
        return z1, h1, z2, h2, logits

    def _probs(self, batch):
        _z1, _h1, _z2, _h2, logits = self._forward(batch)
        return _softmax(logits)

    def sgd_step(self, batch, labels, lr):
        grads, _loss = self.gradients(batch, labels)
        self.W1 -= lr * grads[0]
        self.b1 -= lr * grads[1]
        self.W2 -= lr * grads[2]
        self.b2 -= lr * grads[3]
        self.W3 -= lr * grads[4]
        self.b3 -= lr * grads[5]

    def gradients(self, batch, labels):
        batch = np.asarray(batch, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        count = batch.shape[0]
        z1, h1, z2, h2, logits = self._forward(batch)
        probs = _softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(count), labels] -= 1.0
        dlogits /= count
        picked = np.clip(probs[np.arange(count), labels], 1e-8, 1.0)
        loss = float(-np.mean(np.log(picked)))

        dW3 = dlogits.T @ h2
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3
        dz2 = dh2 * (z2 > 0)

        dW2 = dz2.T @ h1
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (z1 > 0)

        dW1 = dz1.T @ batch
        db1 = dz1.sum(axis=0)
        grads = [
            dW1.astype(np.float32),
            db1.astype(np.float32),
            dW2.astype(np.float32),
            db2.astype(np.float32),
            dW3.astype(np.float32),
            db3.astype(np.float32),
        ]
        return grads, loss


def _read_npz(path):
    try:
        return np.load(path)
    except (OSError, ValueError, EOFError):
        return None


def _weights_ok(data):
    try:
        weights = (
            data["W1"].shape == (H1, IN_DIM)
            and data["b1"].shape == (H1,)
            and data["W2"].shape == (H2, H1)
            and data["b2"].shape == (H2,)
            and data["W3"].shape == (OUTPUTS, H2)
            and data["b3"].shape == (OUTPUTS,)
        )
        if not weights:
            return False
        if "replay_x" in data.files:
            replay = data["replay_x"]
            if replay.ndim != 2 or replay.shape[1] != IN_DIM:
                return False
        return True
    except (KeyError, ValueError):
        return False


def _weights_of(data):
    return (
        data["W1"].astype(np.float32),
        data["b1"].astype(np.float32),
        data["W2"].astype(np.float32),
        data["b2"].astype(np.float32),
        data["W3"].astype(np.float32),
        data["b3"].astype(np.float32),
    )

