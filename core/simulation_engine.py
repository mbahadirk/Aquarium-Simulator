import time


class SimulationEngine:
    def __init__(self, environment, renderer=None, logger=None, config: dict = None):
        self.environment = environment
        self.renderer = renderer
        self.logger = logger
        self.config = config or {}
        self.running = False
        self.paused = False
        self.step_count = 0
        self._fps = self.config.get("simulation", {}).get("fps", 60)
        self._log_interval = self.config.get("logging", {}).get("interval", 100)

    def run(self) -> None:
        self.running = True
        target_dt = 1.0 / self._fps

        while self.running:
            t0 = time.perf_counter()

            if self.renderer:
                events = self.renderer.handle_events()
                if "quit" in events:
                    self.running = False
                    break
                if "pause" in events:
                    self.paused = not self.paused
                if "food_up"    in events: self.environment.increase_food()
                if "food_down"  in events: self.environment.decrease_food()
                if "sfood_up"   in events: self.environment.increase_super_food()
                if "sfood_down" in events: self.environment.decrease_super_food()
                if "pred_up"      in events: self.environment.increase_predators()
                if "pred_down"    in events: self.environment.decrease_predators()
                if "agents_up"    in events: self.environment.increase_agents()
                if "agents_down"  in events: self.environment.decrease_agents()
                if "toggle_debug" in events and self.renderer:
                    self.renderer._debug_vision = not self.renderer._debug_vision

            if not self.paused:
                stats = self.environment.step()
                self.step_count += 1

                if self.logger and self.step_count % self._log_interval == 0:
                    self.logger.log(self.step_count, stats)

            if self.renderer:
                self.renderer.render(self.environment, self.step_count, self.paused)

            elapsed = time.perf_counter() - t0
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._shutdown()

    def stop(self) -> None:
        self.running = False

    def _shutdown(self) -> None:
        if self.renderer:
            self.renderer.close()
        if self.logger:
            self.logger.close()
