from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class World:
    name: str
    facts: dict[str, bool] = field(default_factory=dict)

    def holds(self, fact: str) -> bool:
        return self.facts.get(fact, False)

    def __repr__(self):
        true_facts = [f for f, v in self.facts.items() if v]
        return f"World({self.name}: {true_facts})"


@dataclass
class KripkeModel:
    worlds: dict[str, World] = field(default_factory=dict)
    relations: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    actual_world: str = ""

    # ── world management ──────────────────────────────────────────
    def add_world(self, name: str, facts: dict[str, bool]) -> World:
        w = World(name=name, facts=facts)
        self.worlds[name] = w
        return w

    def set_actual(self, name: str):
        assert name in self.worlds, f"World '{name}' does not exist"
        self.actual_world = name

    # ── relation management ───────────────────────────────────────
    def add_agent(self, agent: str):
        if agent not in self.relations:
            self.relations[agent] = set()

    def make_indistinguishable(self, agent: str, w1: str, w2: str):
        self.relations[agent].add((w1, w2))
        self.relations[agent].add((w2, w1))

    def accessible_worlds(self, agent: str, from_world: str) -> set[str]:
        return {
            w2 for (w1, w2) in self.relations[agent]
            if w1 == from_world
        } | {from_world}

    # ── epistemic queries ─────────────────────────────────────────
    def knows(self, agent: str, fact: str, from_world: str | None = None) -> bool:
        world = from_world or self.actual_world
        reachable = self.accessible_worlds(agent, world)
        return all(self.worlds[w].holds(fact) for w in reachable)

    def believes_possible(self, agent: str, fact: str, from_world: str | None = None) -> bool:
        world = from_world or self.actual_world
        reachable = self.accessible_worlds(agent, world)
        return any(self.worlds[w].holds(fact) for w in reachable)

    def nested_knows(self, agents: list[str], fact: str, from_world: str | None = None) -> bool:
        """
        Evaluates nested knowledge: agents[0] knows that agents[1] knows that ... fact.
        Example: nested_knows(["anne", "sally"], "ball_in_box")
        means: anne knows that sally knows that ball_in_box is true.
        """
        world = from_world or self.actual_world

        if len(agents) == 1:
            return self.knows(agents[0], fact, world)

        outer_agent = agents[0]
        reachable = self.accessible_worlds(outer_agent, world)
        return all(
            self.nested_knows(agents[1:], fact, w)
            for w in reachable
        )

    # ── proof trace ───────────────────────────────────────────────
    def explain(self, agent: str, fact: str, from_world: str | None = None) -> str:
        world = from_world or self.actual_world
        reachable = self.accessible_worlds(agent, world)
        lines = [f"Query: does {agent} know '{fact}' in world '{world}'?"]
        lines.append(f"Worlds accessible to {agent}: {reachable}")
        for w in reachable:
            val = self.worlds[w].holds(fact)
            lines.append(f"  {w}: '{fact}' = {val}")
        result = all(self.worlds[w].holds(fact) for w in reachable)
        lines.append(f"Result: {'YES' if result else 'NO'}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"KripkeModel(\n"
            f"  worlds={list(self.worlds.keys())},\n"
            f"  actual='{self.actual_world}',\n"
            f"  agents={list(self.relations.keys())}\n"
            f")"
        )