"""Window, the floor, and the order you give Wren."""

import pygame

from .brain import Brain
from .settings import (
    ACTION_NAMES,
    BOARD_X,
    BOARD_Y,
    BOTTOM,
    BUTTON,
    BUTTON_EDGE,
    CARD,
    COLS,
    CRACK,
    DANGER,
    DIRS,
    FLOOR_A,
    FLOOR_B,
    FPS,
    GAP,
    GOAL,
    GOAL_DEEP,
    GOLD,
    HEIGHT,
    MAT,
    MAT_EDGE,
    MAX_TURNS,
    MOVE_TIME,
    MUTED,
    PANEL_W,
    PIT_RIM,
    ROWS,
    SPIKE,
    TEXT,
    TILE,
    TITLE,
    WAIT,
    WALL,
    WALL_LIT,
    WIDTH,
)
from .world import GOAL as GOAL_TILE
from .world import HAZARDS, KIND, MAT as MAT_TILE
from .world import WALL as WALL_TILE
from .world import FLOORS, Grid, destination

MOVE_KEYS = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_RIGHT: 1,
    pygame.K_d: 1,
    pygame.K_DOWN: 2,
    pygame.K_s: 2,
    pygame.K_LEFT: 3,
    pygame.K_a: 3,
    pygame.K_SPACE: WAIT,
}


def load_font(size, bold=False):
    return pygame.font.SysFont("segoeui", size, bold=bold)


def wrap(font, text, width):
    lines = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if font.size(trial)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = load_font(22)
        self.small = load_font(16)
        self.tiny = load_font(15)
        self.button_font = load_font(20, bold=True)
        self.big = load_font(42, bold=True)
        self.shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.shade.fill((5, 8, 12, 210))
        self.running = True
        self.notes = []
        self.card_rect = None
        self.play_rect = pygame.Rect(0, 0, 240, 52)
        self.forget_rect = pygame.Rect(0, 0, 240, 52)
        self.quit_rect = pygame.Rect(0, 0, 240, 52)
        try:
            self.brain = Brain.load()
            self.habit = Brain.load_factory()
        except FileNotFoundError as err:
            pygame.quit()
            print(err)
            raise SystemExit(1) from err
        self.flash = 0.0
        self.load_floor(0, fresh=True)
        self.mode = "title"

    def load_floor(self, index, fresh):
        self.floor_index = index
        self.floor_name, self.blurb, rows = FLOORS[index]
        self.grid = Grid(rows)
        self.units = self.grid.units()
        self.graves = []
        self.turns = 0
        self.phase = "input"
        self.mover = None
        self.queue = []
        self.queue_i = 0
        self.anim = 0.0
        self.action = WAIT
        self.player_action = WAIT
        self.origin = (0, 0)
        self.dest = (0, 0)
        self.seen_avoid = set()
        self.fail_reason = ""
        if fresh:
            self.notes = []
            self.session_taught = []
            self.lessons = 0
        self.mode = "intro"
        self.refresh_read()

    def start_run(self):
        self.brain = Brain.load()
        self.habit = Brain.load_factory()
        self.flash = 0.0
        self.load_floor(0, fresh=True)

    def forget(self):
        Brain.forget()
        self.brain = Brain.load()
        self.habit = Brain.load_factory()

    def retry_floor(self):
        for unit in self.units:
            unit.reset()
        self.graves = []
        self.turns = 0
        self.phase = "input"
        self.mover = None
        self.seen_avoid = set()
        self.fail_reason = ""
        self.mode = "play"
        self.note("The floor is set again. The lessons stayed.", MUTED)
        self.refresh_read()

    def advance(self):
        if self.floor_index >= len(FLOORS) - 1:
            self.mode = "won"
            return
        self.load_floor(self.floor_index + 1, fresh=False)

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle()
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.click(event.pos)

    def key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        if self.mode == "title":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.start_run()
            elif key == pygame.K_F2:
                self.forget()
            return
        if self.mode == "intro":
            if key == pygame.K_RETURN:
                self.mode = "play"
                self.refresh_read()
                return
            action = MOVE_KEYS.get(key)
            if action is not None:
                self.mode = "play"
                self.refresh_read()
                self.order(action)
            return
        if self.mode == "cleared":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.advance()
            return
        if self.mode == "failed":
            if key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_r):
                self.retry_floor()
            return
        if self.mode == "won":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.mode = "title"
            return
        if self.phase != "input":
            return
        if key == pygame.K_r:
            self.retry_floor()
            return
        if key == pygame.K_RETURN:
            self.order(self.read_action)
            return
        action = MOVE_KEYS.get(key)
        if action is not None:
            self.order(action)

    def click(self, pos):
        if self.mode == "title":
            if self.play_rect.collidepoint(pos):
                self.start_run()
            elif self.forget_rect.collidepoint(pos):
                self.forget()
            elif self.quit_rect.collidepoint(pos):
                self.running = False
            return
        if self.card_rect is not None and self.card_rect.collidepoint(pos):
            self.key(pygame.K_RETURN)

    def order(self, action):
        leader = self.units[0]
        if not leader.alive:
            return
        occupied = self.occupied(leader)
        legal = self.grid.legal(leader.x, leader.y, occupied)
        if not legal[action]:
            self.note("That way is shut.", MUTED)
            return
        nx, ny = destination(leader.x, leader.y, action)
        tile = self.grid.tile(nx, ny)
        if action != self.read_action and tile not in HAZARDS:
            obs = self.grid.observe(leader.x, leader.y, occupied)
            try:
                self.brain.learn(obs, action)
            except OSError:
                self.note("The lesson could not be saved.", DANGER)
            self.lessons += 1
            self.flash = 1.0
            self.remember(leader)
            self.note("The squad kept that step.", GOLD)
        self.player_action = action
        self.turns += 1
        self.queue = [unit for unit in self.units if unit.alive]
        self.queue_i = 0
        self.phase = "anim"
        self.begin_mover()

    def remember(self, leader):
        for dx, dy in DIRS:
            nx, ny = leader.x + dx, leader.y + dy
            if nx < 0 or ny < 0 or nx >= COLS or ny >= ROWS:
                continue
            kind = KIND.get(self.grid.tile(nx, ny))
            if kind and kind not in self.session_taught:
                self.session_taught.append(kind)

    def begin_mover(self):
        if self.queue_i >= len(self.queue):
            self.finish_round()
            return
        unit = self.queue[self.queue_i]
        self.mover = unit
        occupied = self.occupied(unit)
        legal = self.grid.legal(unit.x, unit.y, occupied)
        if unit.leader:
            action = self.player_action
        else:
            obs = self.grid.observe(unit.x, unit.y, occupied)
            action, _probs = self.brain.choose(obs, legal)
            self.mention(unit, obs, legal, action)
        if not legal[action]:
            action = WAIT
        self.action = action
        self.origin = (float(unit.x), float(unit.y))
        dx, dy = destination(unit.x, unit.y, action)
        self.dest = (float(dx), float(dy))
        self.anim = 0.0
        if action < WAIT:
            unit.face = action

    def mention(self, unit, obs, legal, action):
        old, _probs = self.habit.choose(obs, legal)
        ox, oy = destination(unit.x, unit.y, old)
        nx, ny = destination(unit.x, unit.y, action)
        old_tile = self.grid.tile(ox, oy)
        new_tile = self.grid.tile(nx, ny)
        if old_tile not in HAZARDS or new_tile in HAZARDS:
            return
        kind = KIND[old_tile]
        key = (unit.name, kind)
        if key in self.seen_avoid:
            return
        self.seen_avoid.add(key)
        earlier = [item for item in self.session_taught if item != kind]
        if earlier:
            self.note(f"{unit.name} treated the {kind} like the {earlier[-1]}.", GOLD)
        else:
            self.note(f"{unit.name} kept off the {kind}.", TEXT)

    def finish_round(self):
        self.mover = None
        self.phase = "input"
        living = [unit for unit in self.units if unit.alive]
        if living and all(self.grid.tile(unit.x, unit.y) == GOAL_TILE for unit in living):
            self.mode = "won" if self.floor_index >= len(FLOORS) - 1 else "cleared"
            return
        if self.turns >= MAX_TURNS:
            self.mode = "failed"
            self.fail_reason = "time"
            return
        self.refresh_read()

    def update(self, dt):
        self.flash = max(0.0, self.flash - dt * 1.4)
        if self.mode != "play" or self.phase != "anim" or self.mover is None:
            return
        self.anim += dt
        if self.anim < MOVE_TIME:
            return
        unit = self.mover
        unit.x, unit.y = int(self.dest[0]), int(self.dest[1])
        tile = self.grid.tile(unit.x, unit.y)
        if tile in HAZARDS:
            unit.alive = False
            self.graves.append((unit.x, unit.y, unit.color))
            self.note(f"{unit.name} fell on the {KIND[tile]}.", DANGER)
            if unit.leader:
                self.mode = "failed"
                self.fail_reason = "fell"
                self.phase = "input"
                self.mover = None
                return
        self.queue_i += 1
        self.begin_mover()

    def refresh_read(self):
        leader = self.units[0]
        occupied = self.occupied(leader)
        obs = self.grid.observe(leader.x, leader.y, occupied)
        legal = self.grid.legal(leader.x, leader.y, occupied)
        action, probs = self.brain.choose(obs, legal)
        self.read_action = action
        self.read_probs = probs

    def occupied(self, unit):
        return {(other.x, other.y) for other in self.units if other.alive and other is not unit}

    def note(self, text, color):
        self.notes.append((text, color))
        self.notes = self.notes[-5:]

    def run_pixel(self, unit):
        if self.mover is unit and self.phase == "anim" and self.mode == "play":
            span = max(MOVE_TIME, 0.001)
            t = min(1.0, self.anim / span)
            t = t * t * (3.0 - 2.0 * t)
            x = self.origin[0] + (self.dest[0] - self.origin[0]) * t
            y = self.origin[1] + (self.dest[1] - self.origin[1]) * t
        else:
            x, y = unit.x, unit.y
        return BOARD_X + x * TILE + TILE / 2, BOARD_Y + y * TILE + TILE / 2

    def draw(self):
        self.screen.fill((8, 14, 18))
        self.draw_board()
        if self.mode == "title":
            self.screen.blit(self.shade, (0, 0))
        self.draw_panel()
        self.card_rect = None
        if self.mode == "title":
            self.draw_title()
        elif self.mode == "cleared":
            self.draw_end_card("The door is open.", self.clear_lines())
        elif self.mode == "failed":
            title = "Wren fell." if self.fail_reason == "fell" else "The steps ran out."
            self.draw_end_card(title, ["Enter sets the floor again. The lessons stayed."])
        elif self.mode == "won":
            self.draw_end_card("The squad is through.", self.won_lines())
        pygame.display.flip()

    def draw_board(self):
        for y in range(ROWS):
            for x in range(COLS):
                self.draw_tile(x, y, self.grid.tile(x, y))
        for x, y, color in self.graves:
            rect = self.tile_rect(x, y)
            pygame.draw.line(
                self.screen,
                color,
                (rect.centerx - 8, rect.centery + 4),
                (rect.centerx + 8, rect.centery + 4),
                3,
            )
        if self.mode in ("intro", "play") and self.phase == "input":
            self.draw_ghost()
        for unit in self.units:
            if unit.alive or unit is self.mover:
                self.draw_unit(unit)
        hint = self.hint()
        if hint:
            bar_y = BOARD_Y + ROWS * TILE + 16
            label = self.small.render(hint, True, MUTED)
            self.screen.blit(label, (BOARD_X, bar_y))

    def draw_tile(self, x, y, tile):
        rect = self.tile_rect(x, y)
        if tile == WALL_TILE:
            pygame.draw.rect(self.screen, WALL, rect)
            pygame.draw.line(self.screen, WALL_LIT, rect.topleft, rect.topright, 2)
            return
        base = FLOOR_A if (x + y) % 2 == 0 else FLOOR_B
        pygame.draw.rect(self.screen, base, rect)
        if tile == GOAL_TILE:
            inner = rect.inflate(-8, -6)
            pygame.draw.rect(self.screen, GOAL_DEEP, inner, border_radius=3)
            pygame.draw.rect(self.screen, GOAL, inner.inflate(-8, -14), border_radius=2)
        elif tile in HAZARDS and KIND.get(tile) == "pit":
            rim = rect.inflate(-10, -12)
            pygame.draw.ellipse(self.screen, PIT_RIM, rim)
            pygame.draw.ellipse(self.screen, (5, 7, 10), rim.inflate(-8, -8))
        elif tile in HAZARDS and KIND.get(tile) == "spikes":
            self.draw_spikes(rect)
        elif tile in HAZARDS and KIND.get(tile) == "crack":
            self.draw_crack(rect)
        elif tile == MAT_TILE:
            slab = rect.inflate(-12, -12)
            pygame.draw.rect(self.screen, MAT_EDGE, slab, border_radius=8)
            pygame.draw.rect(self.screen, MAT, slab.inflate(-4, -4), border_radius=6)

    def draw_spikes(self, rect):
        cx, cy = rect.center
        for offset in (-11, 0, 11):
            pygame.draw.polygon(
                self.screen,
                SPIKE,
                [
                    (cx + offset, cy - 10),
                    (cx + offset - 6, cy + 10),
                    (cx + offset + 6, cy + 10),
                ],
            )

    def draw_crack(self, rect):
        cx, cy = rect.center
        color = CRACK
        pygame.draw.lines(
            self.screen,
            color,
            False,
            [(cx - 14, cy - 4), (cx - 4, cy + 2), (cx + 2, cy - 6), (cx + 14, cy + 4)],
            3,
        )
        pygame.draw.lines(
            self.screen,
            color,
            False,
            [(cx - 8, cy + 8), (cx, cy + 1), (cx + 8, cy + 9)],
            2,
        )

    def draw_ghost(self):
        leader = self.units[0]
        if not leader.alive:
            return
        nx, ny = destination(leader.x, leader.y, self.read_action)
        rect = self.tile_rect(nx, ny).inflate(-8, -8)
        tile = self.grid.tile(nx, ny)
        color = DANGER if tile in HAZARDS else GOLD
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=8)

    def draw_unit(self, unit):
        cx, cy = self.run_pixel(unit)
        ix, iy = int(cx), int(cy)
        pygame.draw.ellipse(self.screen, (4, 8, 10), (ix - 10, iy + 8, 20, 8))
        if self.flash > 0 and unit.alive:
            radius = int(20 + (1.0 - self.flash) * 8)
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            alpha = int(170 * self.flash)
            pygame.draw.circle(glow, (255, 214, 120, alpha), (radius, radius), radius, 2)
            self.screen.blit(glow, (ix - radius, iy - radius))
        pygame.draw.circle(self.screen, unit.color, (ix, iy), 13)
        pygame.draw.circle(self.screen, (8, 14, 18), (ix, iy), 13, 2)
        if unit.leader:
            pygame.draw.circle(self.screen, GOLD, (ix, iy), 17, 2)
        face = unit.face if unit.face < 4 else 1
        dx, dy = DIRS[face]
        pygame.draw.circle(self.screen, (8, 14, 18), (ix + dx * 7, iy + dy * 7), 3)

    def draw_panel(self):
        x = BOARD_X + COLS * TILE + GAP
        y = BOARD_Y
        title = self.big.render("Squad", True, TEXT)
        self.screen.blit(title, (x, y))
        y += 52
        name = self.font.render(self.floor_name, True, GOLD)
        self.screen.blit(name, (x, y))
        y += 32
        for line in wrap(self.tiny, self.blurb, PANEL_W - 8):
            self.screen.blit(self.tiny.render(line, True, MUTED), (x, y))
            y += 20
        y += 12
        pygame.draw.line(self.screen, BUTTON_EDGE, (x, y), (x + PANEL_W - 24, y), 1)
        y += 14
        if self.mode in ("intro", "play", "failed", "cleared"):
            line, pct = self.read_text()
            self.screen.blit(self.tiny.render("Wren's read", True, MUTED), (x, y))
            y += 20
            danger = "into the" in line and "door" not in line
            color = DANGER if danger else TEXT
            self.screen.blit(self.font.render(line, True, color), (x, y))
            y += 28
            self.screen.blit(self.tiny.render(f"{pct}%", True, MUTED), (x, y))
            y += 28
        step = self.small.render(f"Orders {self.turns} / {MAX_TURNS}", True, TEXT)
        lessons = self.small.render(f"Lessons {self.lessons}", True, TEXT)
        self.screen.blit(step, (x, y))
        self.screen.blit(lessons, (x + 150, y))
        y += 36
        for unit in self.units:
            color = unit.color if unit.alive else MUTED
            pygame.draw.circle(self.screen, color, (x + 8, y + 8), 6)
            if unit.leader and unit.alive:
                pygame.draw.circle(self.screen, GOLD, (x + 8, y + 8), 9, 1)
            label = unit.name if unit.alive else f"{unit.name} fell"
            self.screen.blit(self.small.render(label, True, color), (x + 24, y - 2))
            y += 26
        y += 8
        pygame.draw.line(self.screen, BUTTON_EDGE, (x, y), (x + PANEL_W - 24, y), 1)
        y += 12
        for text, color in self.notes:
            for line in wrap(self.tiny, text, PANEL_W - 8):
                self.screen.blit(self.tiny.render(line, True, color), (x, y))
                y += 18
            y += 6
        foot = "Lessons stay on this computer."
        self.screen.blit(self.tiny.render(foot, True, MUTED), (x, HEIGHT - BOTTOM + 8))

    def read_text(self):
        action = self.read_action
        leader = self.units[0]
        nx, ny = destination(leader.x, leader.y, action)
        tile = self.grid.tile(nx, ny)
        if action == WAIT:
            where = "hold"
        elif tile in HAZARDS:
            where = f"into the {KIND[tile]}"
        elif tile == GOAL_TILE:
            where = "into the door"
        elif tile == MAT_TILE:
            where = "onto the mat"
        else:
            where = "onto open floor"
        pct = 0 if self.read_probs is None else int(round(float(self.read_probs[action]) * 100))
        return f"{ACTION_NAMES[action]}, {where}", pct

    def hint(self):
        if self.mode == "intro":
            return "Arrows order Wren. Enter stands ready."
        if self.mode != "play" or self.phase != "input":
            return "The squad is moving." if self.phase == "anim" and self.mode == "play" else ""
        _line, _pct = self.read_text()
        nx, ny = destination(self.units[0].x, self.units[0].y, self.read_action)
        if self.grid.tile(nx, ny) in HAZARDS:
            return "Arrows step around. Enter takes the danger. R sets the floor again."
        return "Arrows order Wren. Enter takes the read. R sets the floor again."

    def clear_lines(self):
        gone = [unit.name for unit in self.units if not unit.alive]
        if gone:
            body = "Gone: " + ", ".join(gone) + ". The others made the door."
        else:
            body = "The squad made the door."
        nxt = "Enter" if self.floor_index >= len(FLOORS) - 1 else "Enter for the next floor."
        return [body, nxt]

    def won_lines(self):
        gone = [unit.name for unit in self.units if not unit.alive]
        lines = ["What Wren was taught, the squad kept."]
        if gone:
            lines.append("Gone: " + ", ".join(gone) + ".")
        lines.append("Enter")
        return lines

    def draw_end_card(self, title, lines):
        card = pygame.Rect(0, 0, 460, 150)
        card.center = (BOARD_X + COLS * TILE // 2, BOARD_Y + ROWS * TILE // 2)
        self.card_rect = card
        pygame.draw.rect(self.screen, CARD, card, border_radius=12)
        pygame.draw.rect(self.screen, BUTTON_EDGE, card, 2, border_radius=12)
        self.screen.blit(self.font.render(title, True, GOLD), (card.x + 22, card.y + 18))
        y = card.y + 54
        for line in lines:
            self.screen.blit(self.small.render(line, True, TEXT), (card.x + 22, y))
            y += 26

    def draw_title(self):
        card = pygame.Rect(0, 0, 520, 400)
        card.center = (BOARD_X + COLS * TILE // 2, HEIGHT // 2)
        pygame.draw.rect(self.screen, CARD, card, border_radius=14)
        pygame.draw.rect(self.screen, BUTTON_EDGE, card, 2, border_radius=14)
        self.screen.blit(self.big.render("Squad", True, TEXT), (card.x + 36, card.y + 28))
        y = card.y + 88
        copy = (
            "Four bodies, one brain. You order Wren. "
            "Ash, Bolt, and Nim walk the same weights. "
            "A step you correct is a step they all keep."
        )
        for line in wrap(self.small, copy, card.width - 72):
            self.screen.blit(self.small.render(line, True, MUTED), (card.x + 36, y))
            y += 24
        y += 18
        self.play_rect.center = (card.centerx, y + 10)
        y += 64
        self.forget_rect.center = (card.centerx, y + 10)
        y += 64
        self.quit_rect.center = (card.centerx, y + 10)
        self.draw_button(self.play_rect, "Play")
        self.draw_button(self.forget_rect, "Forget lessons", enabled=Brain.user_path() is not None)
        self.draw_button(self.quit_rect, "Quit")

    def draw_button(self, rect, label, enabled=True):
        hot = enabled and rect.collidepoint(pygame.mouse.get_pos())
        fill = (22, 40, 48) if hot else BUTTON
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, GOLD if hot else BUTTON_EDGE, rect, 2, border_radius=10)
        color = TEXT if enabled else MUTED
        text = self.button_font.render(label, True, color)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def tile_rect(self, x, y):
        return pygame.Rect(BOARD_X + x * TILE + 1, BOARD_Y + y * TILE + 1, TILE - 2, TILE - 2)


def main():
    Game().run()
