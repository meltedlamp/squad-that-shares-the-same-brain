"""Teach the factory brain to walk at the door, pits and all, and write squad/weights.npz."""

import numpy as np

from .brain import IN_DIM, Brain, rotate_action, rotate_obs
from .settings import (
    ACTION_NAMES,
    COLS,
    DIR_FEATURES,
    F_HAZARD,
    MAX_TURNS,
    ROWS,
    WAIT,
)
from .world import (
    CRACK,
    FLOORS,
    GOAL,
    HAZARDS,
    KIND,
    MAT,
    PIT,
    SPIKE,
    WALL,
    Grid,
    clockwise,
    counter,
    destination,
    hazard_ahead,
    naive_action,
    observe,
)


EPOCHS = 28
BATCH = 128
ADAM_LR = 0.004
TRAIN_N = 5200
VAL_N = 700


def _check_turns():
    obs = np.zeros(IN_DIM, dtype=np.float32)
    obs[F_HAZARD] = 1.0
    obs[-3] = 1.0
    obs[-1] = 0.4
    turned = rotate_obs(obs, 1)
    if turned[DIR_FEATURES + F_HAZARD] != 1.0:
        raise RuntimeError("a north danger did not turn to the east")
    if abs(float(turned[-3])) > 1e-5 or abs(float(turned[-2]) - 1.0) > 1e-5:
        raise RuntimeError("the door did not turn with the view")
    if rotate_action(2, 1) != 3 or rotate_action(WAIT, 2) != WAIT:
        raise RuntimeError("a step turned the wrong way")
    back = obs
    for _ in range(4):
        back = rotate_obs(back, 1)
    if not np.allclose(back, obs):
        raise RuntimeError("four turns did not restore the view")


def _check_floors():
    print("spawn reads", flush=True)
    for name, _blurb, rows in FLOORS:
        grid = Grid(rows)
        for unit in grid.units():
            tile, action = hazard_ahead(grid.tiles, unit.x, unit.y)
            if tile is None:
                raise RuntimeError(f"{name}: {unit.name} has a safe first step")
            side = clockwise(action)
            nx, ny = destination(unit.x, unit.y, side)
            if grid.tile(nx, ny) in HAZARDS or grid.tile(nx, ny) == WALL:
                raise RuntimeError(f"{name}: {unit.name} cannot step around")
            print(f"  {name}  {unit.name}  {ACTION_NAMES[action]} into {KIND[tile]}", flush=True)


def _random_map(rng):
    from .world import FLOOR

    tiles = np.full((ROWS, COLS), FLOOR, dtype=np.int64)
    tiles[0, :] = WALL
    tiles[-1, :] = WALL
    tiles[:, 0] = WALL
    tiles[:, -1] = WALL
    for _ in range(int(rng.integers(3, 7))):
        x = int(rng.integers(2, COLS - 4))
        y = int(rng.integers(2, ROWS - 4))
        w = int(rng.integers(1, 4))
        h = int(rng.integers(1, 3))
        tiles[y : y + h, x : x + w] = WALL
    kinds = (PIT, SPIKE, CRACK)
    for _ in range(int(rng.integers(5, 10))):
        x = int(rng.integers(1, COLS - 1))
        y = int(rng.integers(1, ROWS - 1))
        if int(tiles[y, x]) == FLOOR:
            tiles[y, x] = kinds[int(rng.integers(0, 3))]
    for _ in range(3):
        x = int(rng.integers(1, COLS - 1))
        y = int(rng.integers(1, ROWS - 1))
        if int(tiles[y, x]) == FLOOR:
            tiles[y, x] = MAT
    gx = int(rng.integers(COLS - 4, COLS - 1))
    gy = int(rng.integers(1, ROWS - 2))
    tiles[gy : gy + 2, gx] = GOAL
    goals = {(x, y) for y in range(ROWS) for x in range(COLS) if int(tiles[y, x]) == GOAL}
    return tiles, goals


def _harvest(tiles, goals, safe, hazard, repeats=1):
    from .world import FLOOR

    height, width = tiles.shape
    for y in range(height):
        for x in range(width):
            tile = int(tiles[y, x])
            if tile == WALL or tile in HAZARDS:
                continue
            if tile not in (FLOOR, MAT, GOAL):
                continue
            action = naive_action(tiles, x, y, goals)
            sample = observe(tiles, x, y, goals)
            nx, ny = destination(x, y, action)
            enters = action != WAIT and int(tiles[ny, nx]) in HAZARDS
            pool = hazard if enters else safe
            for _ in range(repeats):
                pool.append((sample, action))


def _pools(rng):
    safe, hazard = [], []
    for _ in range(80):
        tiles, goals = _random_map(rng)
        _harvest(tiles, goals, safe, hazard, repeats=1)
    for _name, _blurb, rows in FLOORS:
        grid = Grid(rows)
        _harvest(grid.tiles, grid.goals, safe, hazard, repeats=8)
        for unit in grid.units():
            _harvest_one(grid, unit.x, unit.y, safe, hazard, repeats=40)
    if len(safe) < 200 or len(hazard) < 200:
        raise RuntimeError(f"thin practice set: {len(safe)} safe, {len(hazard)} hazard")
    print(f"practice  safe {len(safe)}  hazard {len(hazard)}", flush=True)
    return safe, hazard


def _harvest_one(grid, x, y, safe, hazard, repeats):
    action = naive_action(grid.tiles, x, y, grid.goals)
    sample = observe(grid.tiles, x, y, grid.goals)
    nx, ny = destination(x, y, action)
    enters = action != WAIT and grid.tile(nx, ny) in HAZARDS
    pool = hazard if enters else safe
    for _ in range(repeats):
        pool.append((sample, action))


def _take(pool, count, rng):
    pick = rng.integers(0, len(pool), size=count)
    data_x = np.stack([pool[int(index)][0] for index in pick]).astype(np.float32)
    data_y = np.array([pool[int(index)][1] for index in pick], dtype=np.int64)
    return data_x, data_y


def _mix(safe, hazard, count, rng, hazard_fraction=0.42):
    hazard_n = int(count * hazard_fraction)
    safe_n = count - hazard_n
    hazard_x, hazard_y = _take(hazard, hazard_n, rng)
    safe_x, safe_y = _take(safe, safe_n, rng)
    return np.concatenate([hazard_x, safe_x]), np.concatenate([hazard_y, safe_y])


def _anchors(safe, rng, count=96):
    take = min(count, len(safe))
    pick = rng.choice(len(safe), size=take, replace=False)
    data_x = np.stack([safe[int(index)][0] for index in pick]).astype(np.float32)
    data_y = np.array([safe[int(index)][1] for index in pick], dtype=np.int64)
    return data_x, data_y


class _Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]

    def step(self, params, grads):
        self.t += 1
        b1_hat = 1 - 0.9**self.t
        b2_hat = 1 - 0.999**self.t
        for index, (param, grad) in enumerate(zip(params, grads)):
            self.m[index] = 0.9 * self.m[index] + 0.1 * grad
            self.v[index] = 0.999 * self.v[index] + 0.001 * (grad * grad)
            param -= self.lr * (self.m[index] / b1_hat) / (np.sqrt(self.v[index] / b2_hat) + 1e-8)


def _jitter(batch, rng):
    out = batch.copy()
    view = out[:, : 4 * DIR_FEATURES].reshape(-1, 4, DIR_FEATURES)
    noise = rng.normal(0, 0.03, (view.shape[0], 4, 2)).astype(np.float32)
    view[:, :, 2:4] += noise
    np.clip(view[:, :, 2], 0, 1, out=view[:, :, 2])
    np.clip(view[:, :, 3], 0, 1, out=view[:, :, 3])
    return out


def fit(brain, train_x, train_y, val_x, val_y, rng):
    params = [brain.W1, brain.b1, brain.W2, brain.b2, brain.W3, brain.b3]
    adam = _Adam(params, ADAM_LR)
    order = np.arange(len(train_y))
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH):
            pick = order[start : start + BATCH]
            batch = _jitter(train_x[pick], rng)
            grads, loss = brain.gradients(batch, train_y[pick])
            adam.step(params, grads)
            total += loss * len(pick)
            seen += len(pick)
        if epoch == 0 or epoch == EPOCHS - 1 or epoch % 5 == 4:
            acc = _accuracy(brain, val_x, val_y)
            print(f"epoch {epoch + 1:02d}  loss {total / seen:.3f}  val {acc:.3f}", flush=True)


def _accuracy(brain, data_x, data_y):
    probs = brain._probs(data_x)
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == data_y))


def _spawn_report(brain):
    print("factory at the spawns", flush=True)
    misses = 0
    for name, _blurb, rows in FLOORS:
        grid = Grid(rows)
        for unit in grid.units():
            occupied = set()
            obs = grid.observe(unit.x, unit.y, occupied)
            legal = grid.legal(unit.x, unit.y, occupied)
            action, _probs = brain.choose(obs, legal)
            nx, ny = destination(unit.x, unit.y, action)
            tile = grid.tile(nx, ny)
            mark = KIND.get(tile, "floor")
            print(f"  {name}  {unit.name}  {ACTION_NAMES[action]}  {mark}", flush=True)
            if tile not in HAZARDS:
                misses += 1
    return misses


def _teacher(brain, grid, unit, occupied):
    obs = grid.observe(unit.x, unit.y, occupied)
    legal = grid.legal(unit.x, unit.y, occupied)
    proposed, _probs = brain.choose(obs, legal)
    nx, ny = destination(unit.x, unit.y, proposed)
    if proposed == WAIT or grid.tile(nx, ny) not in HAZARDS:
        return proposed, obs, False
    for cand in (clockwise(proposed), counter(proposed), WAIT):
        if not legal[cand]:
            continue
        cx, cy = destination(unit.x, unit.y, cand)
        if grid.tile(cx, cy) in HAZARDS:
            continue
        return cand, obs, True
    return WAIT, obs, proposed != WAIT


def _step(unit, action, grid, deaths):
    if action < WAIT:
        unit.face = action
    unit.x, unit.y = destination(unit.x, unit.y, action)
    tile = grid.tile(unit.x, unit.y)
    if tile in HAZARDS:
        unit.alive = False
        deaths.append(f"{unit.name} {KIND[tile]}")


def play_floor(brain, floor, habit):
    name, _blurb, rows = floor
    grid = Grid(rows)
    units = grid.units()
    deaths = []
    corrections = 0
    transfers = 0
    for _turn in range(MAX_TURNS):
        living = [unit for unit in units if unit.alive]
        if living and all(grid.tile(unit.x, unit.y) == GOAL for unit in living):
            return _report(name, "clear", deaths, corrections, transfers)
        leader = units[0]
        if not leader.alive:
            return _report(name, "leader", deaths, corrections, transfers)
        occupied = {(unit.x, unit.y) for unit in units if unit.alive and unit is not leader}
        action, obs, teach = _teacher(brain, grid, leader, occupied)
        if teach:
            brain.learn(obs, action, save=False)
            corrections += 1
        _step(leader, action, grid, deaths)
        if not leader.alive:
            return _report(name, "leader", deaths, corrections, transfers)
        for unit in units[1:]:
            if not unit.alive:
                continue
            occupied = {(other.x, other.y) for other in units if other.alive and other is not unit}
            seen = grid.observe(unit.x, unit.y, occupied)
            legal = grid.legal(unit.x, unit.y, occupied)
            action, _probs = brain.choose(seen, legal)
            old, _old_probs = habit.choose(seen, legal)
            ox, oy = destination(unit.x, unit.y, old)
            nx, ny = destination(unit.x, unit.y, action)
            if grid.tile(ox, oy) in HAZARDS and grid.tile(nx, ny) not in HAZARDS:
                transfers += 1
            _step(unit, action, grid, deaths)
    living = [unit for unit in units if unit.alive]
    if living and all(grid.tile(unit.x, unit.y) == GOAL for unit in living):
        return _report(name, "clear", deaths, corrections, transfers)
    return _report(name, "time", deaths, corrections, transfers)


def _report(name, result, deaths, corrections, transfers):
    fallen = ", ".join(deaths) if deaths else "none"
    print(
        f"  {name}  {result}  corrections {corrections}  "
        f"shared {transfers}  fell {fallen}",
        flush=True,
    )
    return result == "clear" and not deaths


def drill(brain):
    habit = brain.copy()
    playing = brain.copy()
    print("drill", flush=True)
    ok = True
    for lap in (1, 2):
        print(f" lap {lap}", flush=True)
        for floor in FLOORS:
            ok = play_floor(playing, floor, habit) and ok
    return ok


def main():
    _check_turns()
    _check_floors()
    rng = np.random.default_rng(7)
    safe, hazard = _pools(np.random.default_rng(4))
    train_x, train_y = _mix(safe, hazard, TRAIN_N, rng)
    val_x, val_y = _mix(safe, hazard, VAL_N, np.random.default_rng(11), hazard_fraction=0.42)
    print(f"train {len(train_y)}  val {len(val_y)}", flush=True)
    brain = Brain.create(np.random.default_rng(3))
    fit(brain, train_x, train_y, val_x, val_y, np.random.default_rng(5))
    print(f"val { _accuracy(brain, val_x, val_y):.3f}", flush=True)
    anchor_x, anchor_y = _anchors(safe, np.random.default_rng(9))
    brain.anchor_x = anchor_x
    brain.anchor_y = anchor_y
    misses = _spawn_report(brain)
    passed = drill(brain)
    brain.save_factory(anchor_x, anchor_y)
    from .brain import FACTORY

    print(f"saved {FACTORY}")
    if misses:
        print(f"warning: the factory walked safely at {misses} spawns")
    if not passed:
        print("warning: a taught squad still fell or missed a door")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
