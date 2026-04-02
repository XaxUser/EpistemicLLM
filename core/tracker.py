from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from core.kripke import KripkeModel


class EventType(Enum):
    MOVE_OBJECT  = "move_object"
    AGENT_ENTERS = "agent_enters"
    AGENT_LEAVES = "agent_leaves"
    AGENT_LOOKS  = "agent_looks"
    COMMUNICATE  = "communicate"


@dataclass
class Event:
    type: EventType
    actor: str
    witnesses: list[str]
    payload: dict

    @staticmethod
    def move(actor: str, witnesses: list[str], obj: str,
             from_loc: str, to_loc: str) -> Event:
        return Event(
            type=EventType.MOVE_OBJECT,
            actor=actor,
            witnesses=witnesses,
            payload={"object": obj, "from": from_loc, "to": to_loc}
        )

    @staticmethod
    def enters(agent: str, witnesses: list[str]) -> Event:
        return Event(
            type=EventType.AGENT_ENTERS,
            actor=agent,
            witnesses=witnesses,
            payload={}
        )

    @staticmethod
    def leaves(agent: str, witnesses: list[str]) -> Event:
        return Event(
            type=EventType.AGENT_LEAVES,
            actor=agent,
            witnesses=witnesses,
            payload={}
        )

    @staticmethod
    def communicate(speaker: str, listener: str, fact: str, value: bool) -> Event:
        return Event(
            type=EventType.COMMUNICATE,
            actor=speaker,
            witnesses=[listener],
            payload={"fact": fact, "value": value}
        )
    
    @staticmethod
    def looks(agent: str, witnesses: list[str], fact: str, value: bool) -> Event:
        return Event(
            type=EventType.AGENT_LOOKS,
            actor=agent,
            witnesses=witnesses,
            payload={"fact": fact, "value": value}
        )

    def __repr__(self):
        return f"Event({self.type.value}, actor={self.actor}, payload={self.payload})"


@dataclass
class KnowledgeGrant:
    """
    Records that an agent was told a fact has a specific value.
    Used during model rebuild to remove edges to contradicting worlds.
    """
    agent: str
    fact: str
    value: bool


class BeliefTracker:
    """
    Processes a sequence of events and maintains a KripkeModel
    that reflects each agent's current belief state.

    Key design:
    - snapshots[i]           = world state BEFORE transition i
    - transition_events[i]   = the MOVE event that caused transition i
    - knowledge_grants       = permanent list of communicated facts
                               applied during every rebuild
    """

    def __init__(self, agents: list[str], initial_facts: dict[str, bool]):
        self.agents = agents
        self.current_facts: dict[str, bool] = initial_facts.copy()
        self.present: set[str] = set(agents)
        self.model = KripkeModel()

        self.snapshots: list[dict[str, bool]] = []
        self.transition_events: list[Event] = []
        self.event_log: list[Event] = []
        self.knowledge_grants: list[KnowledgeGrant] = []   # ← new

        self._rebuild_model()

    # ── model construction ────────────────────────────────────────

    def _rebuild_model(self):
        self.model = KripkeModel()

        all_states = self.snapshots + [self.current_facts.copy()]

        # deduplicate while preserving order
        unique_states: list[dict] = []
        for s in all_states:
            if s not in unique_states:
                unique_states.append(s)

        for i, state in enumerate(unique_states):
            self.model.add_world(f"w{i}", state.copy())

        self.model.set_actual(f"w{len(unique_states) - 1}")

        for agent in self.agents:
            self.model.add_agent(agent)

        # step 1 — add edges for agents who missed transitions
        for i in range(len(unique_states) - 1):
            causing_event = self._causing_event(i)
            if causing_event is None:
                continue
            for agent in self.agents:
                witnessed = (
                    agent == causing_event.actor or
                    agent in causing_event.witnesses
                )
                if not witnessed:
                    self.model.make_indistinguishable(
                        agent, f"w{i}", f"w{i + 1}"
                    )

        # step 2 — apply all knowledge grants (communication effects)
        # remove edges that lead agent to worlds contradicting what they were told
        for grant in self.knowledge_grants:
            self.model.relations[grant.agent] = {
                (w1, w2)
                for (w1, w2) in self.model.relations[grant.agent]
                if not (
                    w1 == self.model.actual_world and
                    self.model.worlds[w2].holds(grant.fact) != grant.value
                ) and not (
                    w2 == self.model.actual_world and
                    self.model.worlds[w1].holds(grant.fact) != grant.value
                )
            }

    def _causing_event(self, transition_idx: int) -> Event | None:
        if transition_idx < len(self.transition_events):
            return self.transition_events[transition_idx]
        return None

    # ── event processing ──────────────────────────────────────────

    def process(self, event: Event):
        self.event_log.append(event)

        match event.type:

            case EventType.AGENT_LEAVES:
                self.present.discard(event.actor)

            case EventType.AGENT_ENTERS:
                self.present.add(event.actor)

            case EventType.MOVE_OBJECT:
                obj      = event.payload["object"]
                from_loc = event.payload["from"]
                to_loc   = event.payload["to"]

                self.snapshots.append(self.current_facts.copy())
                self.transition_events.append(event)

                self.current_facts[f"{obj}_at_{from_loc}"] = False
                self.current_facts[f"{obj}_at_{to_loc}"]   = True

            case EventType.COMMUNICATE:
                # store permanently — applied in every future rebuild
                self.knowledge_grants.append(KnowledgeGrant(
                    agent=event.witnesses[0],
                    fact=event.payload["fact"],
                    value=event.payload["value"]
                ))
            
            case EventType.AGENT_LOOKS:
                # agent explicitly observes the current state
                # treated as a knowledge grant — removes edges to contradicting worlds
                self.knowledge_grants.append(KnowledgeGrant(
                    agent=event.actor,
                    fact=event.payload["fact"],
                    value=event.payload["value"]
                ))

        self._rebuild_model()

    # ── inspection ────────────────────────────────────────────────

    def snapshot(self) -> str:
        lines = ["── Belief tracker snapshot ──────────────"]
        lines.append(f"Present agents : {self.present}")
        lines.append(f"Actual world   : {self.model.actual_world}")
        lines.append(f"All worlds     :")
        for wname, world in self.model.worlds.items():
            marker = " ← actual" if wname == self.model.actual_world else ""
            lines.append(f"  {wname}: {world.facts}{marker}")
        lines.append(f"Relations      :")
        for agent, edges in self.model.relations.items():
            lines.append(f"  {agent}: {edges}")
        lines.append("─────────────────────────────────────────")
        return "\n".join(lines)