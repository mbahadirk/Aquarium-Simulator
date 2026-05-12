"""Right-side settings & inspector panel drawn over the simulation."""

import math
import pygame

# ── Colours ───────────────────────────────────────────────────────────────────
_BG     = (12,  22,  55, 230)
_HDR    = (18,  35,  85, 255)
_ACCENT = (80, 160, 255)
_TEXT   = (220, 230, 255)
_DIM    = (110, 125, 160)
_BTN    = (35,  70, 130)
_BTN_H  = (55, 100, 180)
_RED    = (210,  55,  55)
_GREEN  = (55,  195,  95)
_GOLD   = (255, 200,  55)

PANEL_W   = 340
CONTENT_Y = 70   # y below which content is scrollable


# ── Control definitions ───────────────────────────────────────────────────────
# (label, env_attr, step, min_v, max_v, fmt, use_set_runtime)
_CONTROLS = [
    ("── BESIN ──",         None, None, None,  None,  None,   False),
    ("Spawn hızı",  "_food_spawn_rate",  0.01,  0.01,  1.0,  "{:.2f}", False),
    ("Max besin",   "_max_food",         5,     10,    400,  "{:d}",   False),
    ("── SÜPER BESİN ──",   None, None, None,  None,  None,   False),
    ("Spawn hızı",  "_sf_spawn_rate",    0.001, 0.0,   0.05, "{:.4f}", False),
    ("Max",         "_max_super_food",   1,     0,     20,   "{:d}",   False),
    ("── PREDATÖR ──",      None, None, None,  None,  None,   False),
    ("Spawn hızı",  "_pred_spawn_rate",  0.001, 0.0,   0.05, "{:.4f}", False),
    ("Max",         "_max_predators",    1,     0,     12,   "{:d}",   False),
    ("Hız",         "_rt_pred_speed",    0.1,   0.2,   8.0,  "{:.1f}", True),
    ("Algı r.",     "_rt_pred_det_r",    10,    60,    500,  "{:.0f}", True),
    ("── AJAN ──",          None, None, None,  None,  None,   False),
    ("Max ajan",    "_max_agents",       10,    10,    300,  "{:d}",   False),
    ("Hız",         "_rt_agent_speed",   0.1,   0.5,   10.0, "{:.1f}", True),
    ("Algı r.",     "_rt_agent_det_r",   10,    50,    500,  "{:.0f}", True),
    ("Enerji azalma","_rt_energy_decay", 0.005, 0.005, 0.3,  "{:.3f}", True),
    ("Üreme eşiği", "_rt_repro_thresh",  10,    80,    400,  "{:.0f}", True),
    ("Üreme maliyeti","_rt_repro_cost",  5,     5,     200,  "{:.0f}", True),
]


class SettingsPanel:
    def __init__(self, screen_w: int, screen_h: int, font, font_b, font_s):
        self._sw   = screen_w
        self._sh   = screen_h
        self._f    = font
        self._fb   = font_b
        self._fs   = font_s
        self.visible   = False
        self._tab      = "settings"   # "settings" | "saved" | "inspect"
        self._scroll   = 0
        self._max_sc   = 0

        # Selected entity
        self._sel      = None
        self._name_buf = ""
        self._name_foc = False

        # Saved-tab
        self._saved_sel = -1

        # Surfaces
        self._surf = pygame.Surface((PANEL_W, screen_h), pygame.SRCALPHA)

        # Collected click-rects (rebuilt each draw)
        self._ctrl_rects  = []   # (minus_r, plus_r, attr, step, mn, mx, rt) — screen-space Y
        self._saved_rects = []   # (row_r, spawn_r, record) — screen-space Y

        # Fixed rects (screen-space, rebuilt each draw)
        self._close_r = pygame.Rect(self._sw - 32, 4, 28, 28)
        self._tab_rects: dict = {}
        self._name_box_r = None
        self._save_btn_r = None

    # ── Public ────────────────────────────────────────────────────────────────

    def select(self, entity) -> None:
        self._sel      = entity
        self._name_buf = getattr(entity, "name", "") or ""
        self._name_foc = False
        self._tab      = "inspect"
        self.visible   = True

    def toggle(self) -> None:
        self.visible = not self.visible
        if self.visible and self._tab == "inspect" and self._sel is None:
            self._tab = "settings"

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, event, environment, memory_manager) -> bool:
        if not self.visible:
            return False

        px = self._sw - PANEL_W  # panel left edge in screen coords

        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if mx >= px:
                self._scroll = max(0, min(self._max_sc, self._scroll - event.y * 18))
                return True

        if event.type == pygame.KEYDOWN and self._name_foc:
            if event.key == pygame.K_RETURN:
                self._name_foc = False
            elif event.key == pygame.K_BACKSPACE:
                self._name_buf = self._name_buf[:-1]
            elif event.unicode and len(self._name_buf) < 24:
                self._name_buf += event.unicode
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if mx < px:
                return False

            # Close button (fixed)
            if self._close_r.collidepoint(mx, my):
                self.visible = False
                return True

            # Tab buttons (fixed)
            for tab_id, r in self._tab_rects.items():
                if r.collidepoint(mx, my):
                    self._tab = tab_id
                    self._scroll = 0
                    return True

            # Name box / save button (fixed, inspect tab)
            if self._name_box_r and self._name_box_r.collidepoint(mx, my):
                self._name_foc = True
                return True
            if self._save_btn_r and self._save_btn_r.collidepoint(mx, my):
                self._do_save(memory_manager, environment)
                return True

            # Scrollable content
            if my < CONTENT_Y:
                return True   # in header area but not a button

            # Convert to content-local coords
            cy = my - CONTENT_Y + self._scroll

            # Controls (settings tab)
            for (mr, pr, attr, step, mn, mx_v, rt) in self._ctrl_rects:
                if mr.collidepoint(0, cy):
                    cur = getattr(environment, attr, mn)
                    nv  = max(mn, cur - step)
                    if rt:
                        environment.set_runtime(attr, nv)
                    else:
                        _is_int = isinstance(cur, int) or fmt_is_int(attr)
                        setattr(environment, attr, int(nv) if _is_int else nv)
                    return True
                if pr.collidepoint(0, cx_dummy := 0):
                    pass
                # Use actual x from screen
            # Redo with proper x
            lx = mx - px
            for (mr, pr, attr, step, mn, mx_v, rt) in self._ctrl_rects:
                mr_s = pygame.Rect(mr.x, mr.y, mr.w, mr.h)
                pr_s = pygame.Rect(pr.x, pr.y, pr.w, pr.h)
                hit_y = cy
                if mr_s.x <= lx <= mr_s.x + mr_s.w and mr_s.y <= hit_y <= mr_s.y + mr_s.h:
                    cur = getattr(environment, attr, mn)
                    nv  = max(mn, cur - step)
                    _apply(environment, attr, nv, rt)
                    return True
                if pr_s.x <= lx <= pr_s.x + pr_s.w and pr_s.y <= hit_y <= pr_s.y + pr_s.h:
                    cur = getattr(environment, attr, mn)
                    nv  = min(mx_v, cur + step)
                    _apply(environment, attr, nv, rt)
                    return True

            # Saved agents (saved tab)
            for (row_r, spawn_r, rec) in self._saved_rects:
                if row_r.x <= lx <= row_r.x + row_r.w and row_r.y <= cy <= row_r.y + row_r.h:
                    self._saved_sel = self._saved_rects.index((row_r, spawn_r, rec))
                if spawn_r.x <= lx <= spawn_r.x + spawn_r.w and spawn_r.y <= cy <= spawn_r.y + spawn_r.h:
                    environment.spawn_from_genome(rec)
                    return True

            return True  # click inside panel, always consume

        return False

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, screen, environment, memory_manager, tick: int) -> None:
        if not self.visible:
            self._draw_toggle_button(screen)
            return

        surf = self._surf
        surf.fill((0, 0, 0, 0))
        pygame.draw.rect(surf, _BG, (0, 0, PANEL_W, self._sh))

        # Header
        pygame.draw.rect(surf, _HDR, (0, 0, PANEL_W, 36))
        surf.blit(self._fb.render("⚙  AYARLAR", True, _TEXT), (10, 9))

        # Close button
        local_close = pygame.Rect(PANEL_W - 32, 4, 28, 28)
        pygame.draw.rect(surf, _RED, local_close, border_radius=5)
        cl = self._fs.render("X", True, (255, 255, 255))
        surf.blit(cl, (local_close.x + (28 - cl.get_width()) // 2, local_close.y + 6))
        self._close_r = pygame.Rect(self._sw - 32, 4, 28, 28)

        # Tabs
        tabs = [("settings", "Ayarlar"), ("saved", "Kaydedilenler"), ("inspect", "Seçili")]
        tw = (PANEL_W - 8) // len(tabs)
        self._tab_rects = {}
        for i, (tid, tlbl) in enumerate(tabs):
            tx = 4 + i * tw
            col = _ACCENT if self._tab == tid else _BTN
            pygame.draw.rect(surf, col, (tx, 40, tw - 3, 24), border_radius=4)
            lbl = self._fs.render(tlbl, True, _TEXT)
            surf.blit(lbl, (tx + (tw - 3) // 2 - lbl.get_width() // 2, 45))
            self._tab_rects[tid] = pygame.Rect(self._sw - PANEL_W + tx, 40, tw - 3, 24)

        # Scrollable content area
        content_surf = pygame.Surface((PANEL_W, 3000), pygame.SRCALPHA)
        content_surf.fill((0, 0, 0, 0))

        self._ctrl_rects  = []
        self._saved_rects = []
        self._name_box_r  = None
        self._save_btn_r  = None

        if self._tab == "settings":
            drawn_h = self._draw_settings(content_surf, environment)
        elif self._tab == "saved":
            drawn_h = self._draw_saved(content_surf, memory_manager)
        else:
            drawn_h = self._draw_inspect(content_surf, memory_manager, environment, tick)

        avail_h = self._sh - CONTENT_Y
        self._max_sc = max(0, drawn_h - avail_h)
        surf.blit(content_surf, (0, CONTENT_Y), (0, self._scroll, PANEL_W, avail_h))

        screen.blit(surf, (self._sw - PANEL_W, 0))
        self._draw_toggle_button(screen)

    def _draw_toggle_button(self, screen) -> None:
        """Always-visible [⚙] button in top-right corner."""
        r = pygame.Rect(self._sw - (PANEL_W + 36 if self.visible else 36), 4, 32, 28)
        pygame.draw.rect(screen, _BTN if not self.visible else _ACCENT, r, border_radius=5)
        lbl = self._fs.render("S", True, _TEXT)
        screen.blit(lbl, (r.x + (32 - lbl.get_width()) // 2, r.y + 6))

    # ── Settings tab ──────────────────────────────────────────────────────────

    def _draw_settings(self, surf, environment) -> int:
        y = 6
        rh = 27

        for (label, attr, step, mn, mx_v, fmt, rt) in _CONTROLS:
            if attr is None:
                # Section header
                pygame.draw.line(surf, _ACCENT, (6, y + 9), (PANEL_W - 6, y + 9), 1)
                hl = self._fs.render(label, True, _ACCENT)
                bx = PANEL_W // 2 - hl.get_width() // 2 - 3
                pygame.draw.rect(surf, (12, 22, 55), (bx, y + 1, hl.get_width() + 6, 15))
                surf.blit(hl, (bx + 3, y + 2))
                y += 20
                continue

            val = getattr(environment, attr, None)
            if val is None:
                continue

            is_int = isinstance(val, int)
            val_str = fmt.format(int(val) if is_int else val)

            surf.blit(self._fs.render(label, True, _DIM), (8, y + 5))

            minus_r = pygame.Rect(PANEL_W - 95, y + 3, 22, 21)
            plus_r  = pygame.Rect(PANEL_W - 28, y + 3, 22, 21)
            pygame.draw.rect(surf, _BTN, minus_r, border_radius=3)
            pygame.draw.rect(surf, _BTN, plus_r,  border_radius=3)
            surf.blit(self._fb.render("-", True, _TEXT), (minus_r.x + 5, minus_r.y + 1))
            surf.blit(self._fb.render("+", True, _TEXT), (plus_r.x + 4,  plus_r.y + 1))

            vl = self._fs.render(val_str, True, _TEXT)
            surf.blit(vl, (PANEL_W - 70, y + 5))

            self._ctrl_rects.append((minus_r, plus_r, attr, step, mn, mx_v, rt))
            y += rh

        return y

    # ── Saved agents tab ──────────────────────────────────────────────────────

    def _draw_saved(self, surf, memory_manager) -> int:
        y = 6
        if memory_manager is None:
            surf.blit(self._fs.render("Hafıza yok.", True, _DIM), (8, y))
            return y + 20

        records = memory_manager.load_best_genomes_full()
        if not records:
            surf.blit(self._fs.render("Kayıtlı ajan bulunamadı.", True, _DIM), (8, y))
            return y + 20

        self._saved_rects = []
        for i, rec in enumerate(records):
            sel = (i == self._saved_sel)
            bg  = (30, 58, 115, 200) if sel else (20, 38, 80, 160)
            row_r  = pygame.Rect(4, y, PANEL_W - 8, 50)
            pygame.draw.rect(surf, bg, row_r, border_radius=4)

            name = rec.get("name") or rec.get("id", "?")[:12]
            gen  = rec.get("generation", 0)
            fit  = rec.get("fitness", 0.0)
            food = rec.get("food_eaten", 0)
            surf.blit(self._fs.render(f"{name}", True, _TEXT), (10, y + 3))
            surf.blit(self._fs.render(f"Gen:{gen}  Fit:{fit:.0f}  Besin:{food}", True, _DIM), (10, y + 19))

            spawn_r = pygame.Rect(PANEL_W - 76, y + 12, 68, 24)
            pygame.draw.rect(surf, _GREEN, spawn_r, border_radius=3)
            sl = self._fs.render("Spawn Et", True, (0, 0, 0))
            surf.blit(sl, (spawn_r.x + spawn_r.w // 2 - sl.get_width() // 2, spawn_r.y + 5))

            self._saved_rects.append((row_r, spawn_r, rec))
            y += 56

        return y

    # ── Inspect tab ───────────────────────────────────────────────────────────

    def _draw_inspect(self, surf, memory_manager, environment, tick: int) -> int:
        from aquarium.fish_agent import FishAgent
        from aquarium.predator  import Predator

        ent = self._sel
        if ent is None or not getattr(ent, "alive", False):
            self._sel = None
            surf.blit(self._fs.render("Bir balığa sol tıkla.", True, _DIM), (8, 10))
            return 30

        y = 8
        rh = 18

        def row(lbl, val, col=_TEXT):
            nonlocal y
            surf.blit(self._fs.render(f"{lbl:<16}{val}", True, col), (8, y))
            y += rh

        if isinstance(ent, FishAgent):
            row("ID",         ent.id[:12])
            row("Soy",        ent.lineage_id[:12])
            row("Nesil",      f"Gen {ent.generation}")
            row("Yaş",        f"{ent.age} adım")
            row("Enerji",     f"{ent.energy:.0f} / {ent.reproduction_threshold:.0f}")
            row("Besin yendi",f"{ent.total_food_eaten}")
            row("Çocuk",      f"{ent.children_count}")
            row("Fitness",    f"{ent.fitness:.1f}")
            row("Sprint?",    "Evet" if getattr(ent, "is_sprinting", False) else "Hayır")
            row("Kritik?",    "Evet" if getattr(ent, "is_critical",  False) else "Hayır",
                col=(220, 60, 60) if getattr(ent, "is_critical", False) else _TEXT)

            y += 6
            surf.blit(self._fs.render("İsim:", True, _DIM), (8, y + 3))
            box_r = pygame.Rect(52, y, PANEL_W - 125, 22)
            pygame.draw.rect(surf, (28, 48, 96) if self._name_foc else (18, 32, 72), box_r, border_radius=3)
            pygame.draw.rect(surf, _ACCENT if self._name_foc else _DIM, box_r, 1, border_radius=3)
            disp = self._name_buf + ("|" if self._name_foc and (tick // 30 % 2 == 0) else "")
            surf.blit(self._fs.render(disp, True, _TEXT), (box_r.x + 4, box_r.y + 4))
            self._name_box_r = pygame.Rect(self._sw - PANEL_W + box_r.x, CONTENT_Y + y - self._scroll, box_r.w, box_r.h)

            save_r = pygame.Rect(PANEL_W - 70, y, 62, 22)
            pygame.draw.rect(surf, _GREEN, save_r, border_radius=3)
            sl = self._fs.render("Kaydet", True, (0, 0, 0))
            surf.blit(sl, (save_r.x + save_r.w // 2 - sl.get_width() // 2, save_r.y + 4))
            self._save_btn_r = pygame.Rect(self._sw - PANEL_W + save_r.x, CONTENT_Y + y - self._scroll, save_r.w, save_r.h)
            y += 30

        else:  # Predator
            row("TÜR",     "Predatör", col=_RED)
            row("ID",      ent.id[:12])
            row("Hız",     f"{ent.speed:.2f}")
            row("Kill",    f"{ent.kills}/{ent.max_kills}")
            row("Algı r.", f"{ent.detection_radius:.0f}")
            row("Yaş",     f"{getattr(ent, '_age', 0)} adım")

        return y

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _do_save(self, memory_manager, environment) -> None:
        if memory_manager and self._sel and getattr(self._sel, "alive", False):
            from aquarium.fish_agent import FishAgent
            if isinstance(self._sel, FishAgent):
                self._sel.name = self._name_buf
                self._sel.compute_fitness()
                memory_manager.save_named_agent(self._sel, self._name_buf, environment.step_count)


# ── Utility ───────────────────────────────────────────────────────────────────

def fmt_is_int(attr: str) -> bool:
    return attr in ("_max_food", "_max_super_food", "_max_predators", "_max_agents")


def _apply(environment, attr: str, val: float, use_set_runtime: bool) -> None:
    is_int = fmt_is_int(attr) or isinstance(getattr(environment, attr, val), int)
    val = int(round(val)) if is_int else val
    if use_set_runtime:
        environment.set_runtime(attr, val)
    else:
        setattr(environment, attr, val)
