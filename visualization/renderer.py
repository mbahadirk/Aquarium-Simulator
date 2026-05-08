import math

from typing import Set

import numpy as np
import pygame

from aquarium.fish_agent import FishAgent, get_lineage_color_map
from aquarium.predator import Predator
from aquarium.food import Food
from aquarium.super_food import SuperFood
from visualization.effects import EffectsManager

# ── Palette ───────────────────────────────────────────────────────────────────
BG_COLOR       = (8,  24,  58)
GRID_COLOR     = (12, 35,  75)
FOOD_COLOR     = (80, 220,  80)
TEXT_COLOR     = (220, 230, 255)
DIM_TEXT       = (100, 115, 150)
PAUSE_COLOR    = (255, 200,  60)
BITTEN_RING    = (255,  80,   0)
BOOST_RING     = (255, 230,  80)
PRED_COLOR     = (220,  50,  50)
PRED_OUTLINE   = (255,  80,  80)
SFOOD_COLOR    = (255, 210,  60)
SFOOD_GLOW     = (255, 255, 160)


class Renderer:
    def __init__(self, width: int, height: int, title: str = "Aquarium Simulation"):
        pygame.init()
        pygame.display.set_caption(title)
        self._screen  = pygame.display.set_mode((width, height))
        self._width   = width
        self._height  = height
        self._clock   = pygame.time.Clock()
        self._font    = pygame.font.SysFont("consolas", 14)
        self._font_b  = pygame.font.SysFont("consolas", 17, bold=True)
        self._font_s  = pygame.font.SysFont("consolas", 12)
        self._effects      = EffectsManager(width, height)
        self._tick         = 0
        self._debug_vision = False   # toggled by D key
        self._vision_surf  = pygame.Surface((width, height), pygame.SRCALPHA)

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_events(self) -> Set[str]:
        events: Set[str] = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events.add("quit")
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_ESCAPE:                          events.add("quit")
                elif k == pygame.K_SPACE:                           events.add("pause")
                # Food
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                                                                    events.add("food_up")
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):     events.add("food_down")
                # Super food
                elif k == pygame.K_RIGHTBRACKET:                    events.add("sfood_up")
                elif k == pygame.K_LEFTBRACKET:                     events.add("sfood_down")
                # Predators
                elif k == pygame.K_p:                               events.add("pred_up")
                elif k == pygame.K_o:                               events.add("pred_down")
                # Agent cap
                elif k == pygame.K_UP:                              events.add("agents_up")
                elif k == pygame.K_DOWN:                            events.add("agents_down")
                # Debug vision
                elif k == pygame.K_d:                               events.add("toggle_debug")
        return events

    # ── Main render ───────────────────────────────────────────────────────────

    def render(self, environment, step: int, paused: bool = False) -> None:
        self._tick += 1
        screen = self._screen
        screen.fill(BG_COLOR)

        self._draw_grid()
        if self._debug_vision:
            self._draw_vision_debug(environment)
        self._draw_trails(environment)          # trails first (behind everything)
        self._draw_entities(environment)
        self._effects.update()
        self._collect_waves(environment)        # register new waves from signalers
        self._effects.draw(screen)              # wave rings on top
        self._draw_hud(environment, step)
        self._draw_lineage_legend(environment)

        if paused:
            lbl = self._font_b.render("PAUSED — SPACE to resume", True, PAUSE_COLOR)
            screen.blit(lbl, (self._width // 2 - lbl.get_width() // 2,
                              self._height // 2 - 10))
        pygame.display.flip()
        self._clock.tick()

    # ── Vision debug (D key) ─────────────────────────────────────────────────

    def _draw_vision_debug(self, environment) -> None:
        """Show each agent's detection circle and lines to visible food."""
        surf = self._vision_surf
        surf.fill((0, 0, 0, 0))

        foods   = [e for e in environment.entities if isinstance(e, (Food, SuperFood)) and e.alive]
        agents  = [e for e in environment.entities if isinstance(e, FishAgent) and e.alive]

        for agent in agents:
            ax, ay = int(agent.position[0]), int(agent.position[1])
            r = int(agent.detection_radius)

            # Faint detection circle
            pygame.draw.circle(surf, (60, 120, 255, 35), (ax, ay), r)
            pygame.draw.circle(surf, (80, 150, 255, 80), (ax, ay), r, 1)

            # Eat radius inner circle
            pygame.draw.circle(surf, (80, 220, 80, 60), (ax, ay), int(agent.eat_radius), 1)

            # Lines to each visible food item
            for food in foods:
                dist = agent.distance_to(food)
                if dist < agent.detection_radius:
                    fx, fy = int(food.position[0]), int(food.position[1])
                    # Brighter line when closer
                    intensity = int(120 * (1 - dist / agent.detection_radius))
                    is_super = isinstance(food, SuperFood)
                    col = (intensity, 200, intensity, intensity) if not is_super \
                          else (200, 180, 0, intensity)
                    pygame.draw.line(surf, col, (ax, ay), (fx, fy), 1)

        self._screen.blit(surf, (0, 0))

        # Debug label
        lbl = self._font_s.render("[D] Vision Debug ON", True, (100, 180, 255))
        self._screen.blit(lbl, (self._width // 2 - lbl.get_width() // 2, self._height - 18))

    # ── Grid ─────────────────────────────────────────────────────────────────

    def _draw_grid(self) -> None:
        for x in range(0, self._width, 80):
            pygame.draw.line(self._screen, GRID_COLOR, (x, 0), (x, self._height))
        for y in range(0, self._height, 80):
            pygame.draw.line(self._screen, GRID_COLOR, (0, y), (self._width, y))

    # ── Trail rendering ───────────────────────────────────────────────────────

    def _draw_trails(self, environment) -> None:
        for entity in environment.entities:
            if not (isinstance(entity, FishAgent) and entity.alive):
                continue
            trail = list(entity._trail)
            n = len(trail)
            if n < 2:
                continue

            boosted    = entity.speed_boost_remaining > 0
            sprinting  = getattr(entity, "is_sprinting", False)
            base_r     = entity.RADIUS * 0.55

            for i, (tx, ty, spd) in enumerate(trail):
                age_ratio = (i + 1) / (n + 1)

                scale = 1.0
                if boosted:
                    scale = 1.5
                elif sprinting:
                    scale = 1.25
                r = max(1, int(base_r * age_ratio * scale * max(spd, 0.2)))

                if boosted:
                    blend = age_ratio
                    col = (
                        int(entity.color[0] * (1 - blend) + 255 * blend),
                        int(entity.color[1] * (1 - blend) + 230 * blend),
                        int(entity.color[2] * (1 - blend) +  60 * blend),
                    )
                elif sprinting:
                    # Cyan-white tint for sprint trail
                    blend = age_ratio
                    col = (
                        int(entity.color[0] * (1 - blend) + 80  * blend),
                        int(entity.color[1] * (1 - blend) + 220 * blend),
                        int(entity.color[2] * (1 - blend) + 255 * blend),
                    )
                else:
                    dim = age_ratio * 0.55
                    col = tuple(max(8, int(c * dim)) for c in entity.color)

                pygame.draw.circle(self._screen, col, (int(tx), int(ty)), r)

    # ── Entity rendering ──────────────────────────────────────────────────────

    def _draw_entities(self, environment) -> None:
        for e in environment.entities:
            if isinstance(e, Food)      and e.alive: self._draw_food(e)
        for e in environment.entities:
            if isinstance(e, SuperFood) and e.alive: self._draw_super_food(e)
        for e in environment.entities:
            if isinstance(e, FishAgent) and e.alive: self._draw_agent(e)
        for e in environment.entities:
            if isinstance(e, Predator)  and e.alive: self._draw_predator(e)

    def _draw_food(self, food: Food) -> None:
        x, y = int(food.position[0]), int(food.position[1])
        pygame.draw.circle(self._screen, FOOD_COLOR,        (x, y), food.RADIUS)
        pygame.draw.circle(self._screen, (140, 255, 140),   (x, y), food.RADIUS - 2)

    def _draw_super_food(self, sf: SuperFood) -> None:
        x, y = int(sf.position[0]), int(sf.position[1])
        pr   = sf.pulse_radius(self._tick)
        # Outer glow ring
        pygame.draw.circle(self._screen, (200, 160,  30), (x, y), pr + 3, 2)
        # Body
        pygame.draw.circle(self._screen, SFOOD_COLOR,      (x, y), sf.RADIUS)
        pygame.draw.circle(self._screen, SFOOD_GLOW,       (x, y), sf.RADIUS - 3)
        # Star sparkle — four small lines
        for angle in (0, 90, 45, 135):
            a = math.radians(angle + self._tick * 2)
            x1 = int(x + math.cos(a) * (sf.RADIUS + 5))
            y1 = int(y + math.sin(a) * (sf.RADIUS + 5))
            x2 = int(x + math.cos(a + math.pi) * (sf.RADIUS + 5))
            y2 = int(y + math.sin(a + math.pi) * (sf.RADIUS + 5))
            pygame.draw.line(self._screen, (255, 255, 200), (x1, y1), (x2, y2), 1)

    def _draw_agent(self, agent: FishAgent) -> None:
        x, y = int(agent.position[0]), int(agent.position[1])
        r    = agent.RADIUS

        # Outer rings (bitten = orange, boosted = gold, sprinting = cyan)
        if agent.hits > 0:
            pygame.draw.circle(self._screen, BITTEN_RING, (x, y), r + 4)
        if agent.speed_boost_remaining > 0:
            pygame.draw.circle(self._screen, BOOST_RING,  (x, y), r + 2)
        elif getattr(agent, "is_sprinting", False):
            pygame.draw.circle(self._screen, (80, 220, 255), (x, y), r + 2, 1)

        # Body (lineage colour; dim when critical energy)
        body_col = agent.color
        if getattr(agent, "is_critical", False):
            pulse = abs(math.sin(self._tick / 8))
            body_col = (
                min(255, int(agent.color[0] * 0.5 + 180 * pulse)),
                min(255, int(agent.color[1] * 0.4)),
                min(255, int(agent.color[2] * 0.4)),
            )
        pygame.draw.circle(self._screen, body_col, (x, y), r)

        # Border: white = healthy, dark-orange = bitten, red = critical
        if agent.hits > 0:
            border = (180, 80, 30)
        elif getattr(agent, "is_critical", False):
            border = (220, 40, 40)
        else:
            border = (255, 255, 255)
        pygame.draw.circle(self._screen, border, (x, y), r, 1)

        # Energy bar
        bar_w = r * 2 + 4
        ratio = min(agent.energy / agent.reproduction_threshold, 1.0)
        fill  = int(bar_w * ratio)
        bx, by = x - bar_w // 2, y - r - 7
        pygame.draw.rect(self._screen, (50, 50, 50),  (bx, by, bar_w, 3))
        if fill > 0:
            if getattr(agent, "is_critical", False):
                bar_col = (220, 40, 40)
            elif ratio >= 1.0:
                bar_col = (255, 220, 0)
            else:
                bar_col = (80, 220, 80)
            pygame.draw.rect(self._screen, bar_col, (bx, by, fill, 3))

        # Heading dot
        spd = float(np.linalg.norm(agent.velocity)) + 1e-8
        if spd > 0.1:
            vx, vy = agent.velocity[0] / spd, agent.velocity[1] / spd
            pygame.draw.circle(self._screen, (255, 255, 255),
                               (int(x + vx * (r + 3)), int(y + vy * (r + 3))), 2)

        # Danger signal indicator — small pulsing cyan dot above fish
        if agent.is_signaling:
            pulse = 3 + int(abs(math.sin(self._tick / 5)) * 2)
            pygame.draw.circle(self._screen, (100, 220, 255), (x, y - r - 10), pulse)

        # Mating-ready indicator — pulsing magenta ring below fish
        if agent.can_reproduce():
            pr = 3 + int(abs(math.sin(self._tick / 12)) * 2)
            pygame.draw.circle(self._screen, (255, 80, 200), (x, y + r + 6), pr)

    def _draw_predator(self, pred: Predator) -> None:
        x, y = int(pred.position[0]), int(pred.position[1])
        r    = pred.RADIUS
        spd  = float(np.linalg.norm(pred.velocity)) + 1e-8
        angle = math.atan2(pred.velocity[1], pred.velocity[0]) if spd > 0.05 else 0.0

        def _pt(offset):
            a = angle + offset
            return (int(x + math.cos(a) * r), int(y + math.sin(a) * r))

        pts = [_pt(0), _pt(2.4), _pt(-2.4)]
        pygame.draw.polygon(self._screen, PRED_COLOR,   pts)
        pygame.draw.polygon(self._screen, PRED_OUTLINE, pts, 2)

        lbl = self._font_s.render(f"{pred.kills}/{pred.max_kills}", True, (255, 200, 200))
        self._screen.blit(lbl, (x - lbl.get_width() // 2, y - r - 14))

    # ── Wave management ───────────────────────────────────────────────────────

    def _collect_waves(self, environment) -> None:
        for e in environment.entities:
            if isinstance(e, FishAgent) and e._signal_just_emitted:
                self._effects.add_wave(e.position[0], e.position[1])

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, environment, step: int) -> None:
        agents    = environment.get_entities_of_type(FishAgent)
        predators = environment.get_entities_of_type(Predator)
        foods     = environment.get_entities_of_type(Food)
        sfoods    = environment.get_entities_of_type(SuperFood)
        stats     = environment._collect_stats()
        fps       = self._clock.get_fps()

        bitten    = sum(1 for a in agents if a.hits > 0)
        boosted   = sum(1 for a in agents if a.speed_boost_remaining > 0)
        sprinting = sum(1 for a in agents if getattr(a, "is_sprinting", False))
        critical  = sum(1 for a in agents if getattr(a, "is_critical",  False))
        max_gen   = max((a.generation for a in agents), default=0)

        # Average food items visible per agent
        all_food = [e for e in environment.entities
                    if isinstance(e, (Food, SuperFood)) and e.alive]
        if agents and all_food:
            avg_seen = float(np.mean([
                sum(1 for f in all_food if a.distance_to(f) < a.detection_radius)
                for a in agents
            ]))
        else:
            avg_seen = 0.0

        debug_tag = "  [D ON]" if self._debug_vision else "  [D]"

        rows = [
            ("Step",         f"{step}"),
            ("FPS",          f"{fps:.1f}"),
            ("Agents",       f"{len(agents)} / {stats['agent_max']}  ↑/↓"),
            (" bitten",      f"{bitten}"),
            (" boosted",     f"{boosted}"),
            (" sprinting",   f"{sprinting}"),
            (" critical",    f"{critical}"),
            ("Predators",    f"{len(predators)} / {stats['pred_max']}  P/O"),
            ("Food",         f"{len(foods)} / {stats['food_max']}  +/-"),
            ("SuperFood",    f"{len(sfoods)} / {stats['sf_max']}  ]/["),
            ("Max Gen",      f"{max_gen}"),
            ("Avg seen",     f"{avg_seen:.1f} food/agent{debug_tag}"),
        ]

        pw = 260
        ph = len(rows) * 17 + 12
        surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(surf, (10, 20, 50, 200), (0, 0, pw, ph), border_radius=7)
        for i, (label, val) in enumerate(rows):
            c = (255, 160, 60) if label.startswith(" ") else TEXT_COLOR
            txt = self._font_s.render(f"{label:<12}{val}", True, c)
            surf.blit(txt, (8, 6 + i * 17))
        self._screen.blit(surf, (8, 8))

        hint = self._font_s.render(
            "SPACE:pause  ESC:quit  ↑/↓:agents  P/O:predators  +/-:food  ]/[:superfood  D:vision",
            True, DIM_TEXT,
        )
        self._screen.blit(hint, (8, self._height - 18))

    # ── Lineage legend (top-right) ────────────────────────────────────────────

    def _draw_lineage_legend(self, environment) -> None:
        agents = environment.get_entities_of_type(FishAgent)
        if not agents:
            return
        color_map = get_lineage_color_map()

        # Group by lineage: track max_gen and count
        lineage_data: dict = {}
        for a in agents:
            lid = a.lineage_id
            if lid not in lineage_data:
                lineage_data[lid] = {"max_gen": 0, "count": 0}
            if a.generation > lineage_data[lid]["max_gen"]:
                lineage_data[lid]["max_gen"] = a.generation
            lineage_data[lid]["count"] += 1

        # Sort by highest generation descending, take top 8
        top = sorted(lineage_data.items(), key=lambda kv: kv[1]["max_gen"], reverse=True)[:8]
        if not top:
            return

        entry_h = 18
        pw      = 190
        ph      = len(top) * entry_h + 22
        surf    = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(surf, (10, 20, 50, 200), (0, 0, pw, ph), border_radius=7)

        title = self._font_s.render("NESIL SIRALAMA", True, TEXT_COLOR)
        surf.blit(title, (8, 4))

        for i, (lid, data) in enumerate(top):
            col = color_map.get(lid, (200, 200, 200))
            ey  = 20 + i * entry_h
            pygame.draw.circle(surf, col, (14, ey + 7), 6)
            pygame.draw.circle(surf, (255, 255, 255), (14, ey + 7), 6, 1)
            lbl = self._font_s.render(
                f"Gen{data['max_gen']:<4} {lid[:6]}  ×{data['count']}",
                True, TEXT_COLOR,
            )
            surf.blit(lbl, (25, ey + 1))

        self._screen.blit(surf, (self._width - pw - 8, 8))

    def close(self) -> None:
        pygame.quit()
