import numpy as np
from typing import List

from core.environment import Environment
from aquarium.food import Food
from aquarium.super_food import SuperFood
from aquarium.predator import Predator
from aquarium.fish_agent import FishAgent
from brain.neural_brain import NeuralBrain
from genetics.genome import Genome
from genetics.crossover import crossover
from genetics.mutation import mutate


class AquariumEnvironment(Environment):
    def __init__(self, config: dict, memory_manager=None, lineage_tracker=None):
        wc = config.get("world", {})
        super().__init__(wc.get("width", 1280), wc.get("height", 720))
        self.config          = config
        self.memory_manager  = memory_manager
        self.lineage_tracker = lineage_tracker

        fc  = config.get("food",       {})
        sfc = config.get("super_food", {})
        pc  = config.get("predator",   {})
        ac  = config.get("agent",      {})
        mc  = config.get("memory",     {})

        self._food_spawn_rate:    float = fc.get("spawn_rate",    0.30)
        self._max_food:           int   = fc.get("max_count",      120)
        self._sf_spawn_rate:      float = sfc.get("spawn_rate",  0.005)
        self._max_super_food:     int   = sfc.get("max_count",      5)
        self._pred_spawn_rate:    float = pc.get("spawn_rate",   0.003)
        self._max_predators:      int   = pc.get("max_count",       2)
        self._max_agents:         int   = ac.get("max_count",      120)
        self._min_agents:         int   = ac.get("min_population",   8)
        self._crossover_method:   str   = config.get("genetics", {}).get("crossover_type", "uniform")
        self._save_interval:      int   = mc.get("save_interval",  200)
        self._bite_penalty:       float = ac.get("bite_penalty",  65.0)
        self._bite_cooldown:      int   = ac.get("bite_cooldown_steps", 80)
        self._signal_range:           float = ac.get("signal_range",               150.0)
        self._signal_flee_force:      float = ac.get("signal_flee_force",            0.4)
        self._repro_energy_reward:    float = ac.get("reproduction_energy_reward",  30.0)

        bc = config.get("brain", {})
        self._brain_input  = bc.get("input_size",   28)
        self._brain_hidden = bc.get("hidden_sizes", [16, 8])
        self._brain_output = bc.get("output_size",   2)
        self._genome_size  = NeuralBrain.genome_size(
            self._brain_input, self._brain_hidden, self._brain_output
        )

    # ── Runtime controls ─────────────────────────────────────────────────────

    def increase_food(self,       n: int = 10) -> None: self._max_food       = min(self._max_food + n, 400)
    def decrease_food(self,       n: int = 10) -> None: self._max_food       = max(self._max_food - n, 10)
    def increase_super_food(self, n: int = 1)  -> None: self._max_super_food = min(self._max_super_food + n, 20)
    def decrease_super_food(self, n: int = 1)  -> None: self._max_super_food = max(self._max_super_food - n, 0)
    def increase_predators(self,  n: int = 1)  -> None: self._max_predators  = min(self._max_predators + n, 12)
    def decrease_predators(self,  n: int = 1)  -> None: self._max_predators  = max(self._max_predators - n, 0)
    def increase_agents(self,     n: int = 10) -> None: self._max_agents     = min(self._max_agents + n, 300)
    def decrease_agents(self,     n: int = 10) -> None: self._max_agents     = max(self._max_agents - n, self._min_agents)

    # ── Population seeding ───────────────────────────────────────────────────

    def seed_population(self, saved_genomes: List[dict] = None) -> None:
        ac = self.config.get("agent", {})
        n  = ac.get("initial_count", 30)

        for i in range(n):
            if saved_genomes and i < len(saved_genomes):
                g = saved_genomes[i]
                if len(g["weights"]) != self._genome_size:
                    genome, gen, lid = Genome.random(self._genome_size), 0, None
                else:
                    genome = mutate(Genome.from_list(g["weights"]), self.config)
                    gen    = g.get("generation", 0)
                    lid    = g.get("lineage_id")
            else:
                genome, gen, lid = Genome.random(self._genome_size), 0, None

            agent = self._make_agent(
                position=np.array([
                    np.random.uniform(50, self.width  - 50),
                    np.random.uniform(50, self.height - 50),
                ]),
                genome=genome, generation=gen, parent_ids=[], lineage_id=lid,
            )
            self.add_entity(agent)
            if self.lineage_tracker:
                self.lineage_tracker.register_birth(agent, self.step_count)

        # Pre-seed food spread across the full screen
        for _ in range(min(50, self._max_food)):
            self._spawn_food_random()

    # ── Spawning ─────────────────────────────────────────────────────────────

    def spawn_food(self) -> None:
        self._spawn_food_random()

    def _spawn_food_random(self) -> None:
        x = np.random.uniform(20, self.width  - 20)
        y = np.random.uniform(20, self.height - 20)
        self.add_entity(Food(np.array([x, y]), self.config))

    def spawn_super_food(self) -> None:
        x = np.random.uniform(40, self.width  - 40)
        y = np.random.uniform(40, self.height - 40)
        self.add_entity(SuperFood(np.array([x, y]), self.config))

    def spawn_predator(self) -> None:
        edge = np.random.choice(["left", "right", "bottom"])
        if edge == "left":
            pos = np.array([0.0,             np.random.uniform(0, self.height)])
        elif edge == "right":
            pos = np.array([float(self.width), np.random.uniform(0, self.height)])
        else:
            pos = np.array([np.random.uniform(0, self.width), float(self.height)])
        self.add_entity(Predator(pos, self.config))

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self) -> dict:
        self.step_count += 1

        # 1. Spawning
        if np.random.rand() < self._food_spawn_rate \
                and len(self.get_entities_of_type(Food)) < self._max_food:
            self.spawn_food()
        if np.random.rand() < self._sf_spawn_rate \
                and len(self.get_entities_of_type(SuperFood)) < self._max_super_food:
            self.spawn_super_food()
        if np.random.rand() < self._pred_spawn_rate \
                and len(self.get_entities_of_type(Predator)) < self._max_predators:
            self.spawn_predator()

        # 2. Update entities
        for entity in list(self.entities):
            if entity.alive:
                entity.update(self)

        # 3. Agents eat regular food
        agents = self.get_entities_of_type(FishAgent)
        for agent in agents:
            for food in self.get_entities_of_type(Food):
                if food.alive and agent.distance_to(food) < agent.eat_radius:
                    agent.energy           += food.energy_value
                    agent.total_food_eaten += 1
                    food.alive = False
                    break

        # 4. Agents eat super food (speed boost)
        for agent in agents:
            for sf in self.get_entities_of_type(SuperFood):
                if sf.alive and agent.distance_to(sf) < agent.eat_radius:
                    agent.energy                 += sf.energy_value
                    agent.total_food_eaten       += 1
                    agent.speed_boost_remaining   = sf.speed_boost_duration
                    agent.speed_boost_amount      = sf.speed_boost_amount
                    sf.alive = False
                    break

        # 5. Predators bite agents — 2-hit system
        for pred in self.get_entities_of_type(Predator):
            for agent in list(self.get_entities_of_type(FishAgent)):
                if not agent.alive or pred.distance_to(agent) >= pred.kill_radius:
                    continue
                if agent.hits == 0:
                    agent.energy       -= self._bite_penalty
                    agent.hits          = 1
                    agent.bite_cooldown = self._bite_cooldown
                    if agent.energy <= 0:
                        agent.alive = False
                        agent.compute_fitness()
                        pred.kills += 1
                        if self.lineage_tracker:
                            self.lineage_tracker.register_death(agent, self.step_count, "predation_1hit")
                elif agent.bite_cooldown == 0:
                    agent.alive = False
                    agent.compute_fitness()
                    pred.kills += 1
                    if self.lineage_tracker:
                        self.lineage_tracker.register_death(agent, self.step_count, "predation_2hit")

        # 6. Danger signal propagation (rule-based velocity nudge)
        agents = self.get_entities_of_type(FishAgent)
        signaling = [a for a in agents if a.is_signaling]
        if signaling:
            preds = self.get_entities_of_type(Predator)
            for receiver in agents:
                for sig in signaling:
                    if sig.id == receiver.id:
                        continue
                    if receiver.distance_to(sig) > self._signal_range:
                        continue
                    # Nudge receiver away from the nearest predator
                    if preds:
                        nearest_pred = min(preds, key=lambda p: sig.distance_to(p))
                        flee = receiver.position - nearest_pred.position
                        dist = np.linalg.norm(flee)
                        if dist > 1e-6:
                            receiver.velocity += (flee / dist) * self._signal_flee_force
                    break  # one signal nudge per step is enough

        # 7. Reproduction
        new_agents = self._handle_reproduction()

        # 8. Remove dead entities
        dead = self.remove_dead_entities()
        for e in dead:
            if isinstance(e, FishAgent) and self.lineage_tracker:
                if not self.lineage_tracker.is_registered_dead(e.id):
                    e.compute_fitness()
                    self.lineage_tracker.register_death(e, self.step_count, "starvation")

        # 9. Add offspring
        agents = self.get_entities_of_type(FishAgent)
        for child in new_agents:
            if len(agents) < self._max_agents:
                self.add_entity(child)
                agents.append(child)
                if self.lineage_tracker:
                    self.lineage_tracker.register_birth(child, self.step_count)

        # 10. Population recovery
        self._recover_population()

        # 11. Periodic memory save
        if self.memory_manager and self.step_count % self._save_interval == 0:
            live = self.get_entities_of_type(FishAgent)
            for a in live:
                a.compute_fitness()
            self.memory_manager.save_best_agents(live, self.step_count)

        return self._collect_stats()

    # ── Reproduction ─────────────────────────────────────────────────────────

    def _handle_reproduction(self) -> List[FishAgent]:
        agents     = self.get_entities_of_type(FishAgent)
        candidates = [a for a in agents if a.can_reproduce()]
        already_mated: set    = set()
        offspring:  List[FishAgent] = []

        for p1 in candidates:
            if p1.id in already_mated:
                continue
            partners = [
                a for a in candidates
                if a.id != p1.id and a.id not in already_mated
                and p1.distance_to(a) < p1.detection_radius * 0.6
            ]
            if not partners:
                continue
            p2 = min(partners, key=lambda a: p1.distance_to(a))

            cg1, cg2 = crossover(p1.genome, p2.genome, self._crossover_method)
            cg1 = mutate(cg1, self.config)
            cg2 = mutate(cg2, self.config)
            next_gen = max(p1.generation, p2.generation) + 1

            for cg in [cg1, cg2]:
                offspring.append(self._make_agent(
                    position   = p1.position + np.random.randn(2) * 15,
                    genome     = cg,
                    generation = next_gen,
                    parent_ids = [p1.id, p2.id],
                    lineage_id = p1.lineage_id,
                ))

            p1.energy -= p1.reproduction_cost - self._repro_energy_reward
            p2.energy -= p2.reproduction_cost - self._repro_energy_reward
            p1.children_count += 1;             p2.children_count += 1
            p1._steps_since_last_reproduction = 0
            p2._steps_since_last_reproduction = 0
            already_mated.add(p1.id);  already_mated.add(p2.id)

        return offspring

    # ── Population recovery ───────────────────────────────────────────────────

    def _recover_population(self) -> None:
        agents = self.get_entities_of_type(FishAgent)
        deficit = self._min_agents - len(agents)
        if deficit <= 0:
            return
        saved = self.memory_manager.load_best_genomes() if self.memory_manager else []
        for i in range(deficit):
            if saved:
                g = saved[i % len(saved)]
                if len(g["weights"]) != self._genome_size:
                    genome, gen, lid = Genome.random(self._genome_size), 0, None
                else:
                    genome = mutate(Genome.from_list(g["weights"]), self.config)
                    gen    = g.get("generation", 0)
                    lid    = g.get("lineage_id")
            else:
                genome, gen, lid = Genome.random(self._genome_size), 0, None

            agent = self._make_agent(
                position=np.array([
                    np.random.uniform(60, self.width  - 60),
                    np.random.uniform(60, self.height - 60),
                ]),
                genome=genome, generation=gen, parent_ids=[], lineage_id=lid,
            )
            self.add_entity(agent)
            if self.lineage_tracker:
                self.lineage_tracker.register_birth(agent, self.step_count)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_agent(self, position, genome, generation, parent_ids, lineage_id) -> FishAgent:
        brain = NeuralBrain.from_genome(
            genome, self._brain_input, self._brain_hidden, self._brain_output
        )
        pos = np.clip(
            position,
            [FishAgent.RADIUS, FishAgent.RADIUS],
            [self.width - FishAgent.RADIUS, self.height - FishAgent.RADIUS],
        )
        return FishAgent(
            position=pos, brain=brain, genome=genome, config=self.config,
            generation=generation, parent_ids=parent_ids, lineage_id=lineage_id,
        )

    def _collect_stats(self) -> dict:
        agents = self.get_entities_of_type(FishAgent)
        if agents:
            avg_energy  = float(np.mean([a.energy     for a in agents]))
            avg_fitness = float(np.mean([a.fitness     for a in agents]))
            max_gen     = max(a.generation              for a in agents)
        else:
            avg_energy = avg_fitness = max_gen = 0

        return {
            "step":           self.step_count,
            "population":     len(agents),
            "food_count":     len(self.get_entities_of_type(Food)),
            "sf_count":       len(self.get_entities_of_type(SuperFood)),
            "predator_count": len(self.get_entities_of_type(Predator)),
            "avg_energy":     avg_energy,
            "avg_fitness":    avg_fitness,
            "max_generation": max_gen,
            "food_max":       self._max_food,
            "sf_max":         self._max_super_food,
            "pred_max":       self._max_predators,
            "agent_max":      self._max_agents,
        }
