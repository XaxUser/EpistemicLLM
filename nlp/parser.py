from __future__ import annotations
import json
import os
from openai import OpenAI
from nlp.templates import STORY_EXTRACTION_PROMPT
from core.tracker import BeliefTracker, Event, EventType
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _call_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    Single LLM call. temperature=0 for deterministic extraction.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()



def _parse_json_safe(raw: str) -> dict:
    """
    Strips markdown fences if present, fixes common LLM JSON issues,
    then parses.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    # fix trailing commas before } or ] — common LLM mistake
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return json.loads(raw)


def extract_story_structure(story: str) -> dict:
    """
    Calls LLM to extract structured representation of a story.
    Returns a dict with agents, objects, locations, events, question, answer.
    """
    prompt = STORY_EXTRACTION_PROMPT.format(story=story)
    raw = _call_llm(prompt)
    return _parse_json_safe(raw)


def build_query_from_tracker(structure: dict, tracker: BeliefTracker) -> dict:
    """
    Construit la requête formelle directement depuis le tracker.
    Zéro appel LLM — lecture directe de current_facts et du modèle de Kripke.
    """
    from core.reasoner import Reasoner, Formula

    # agent — premier agent mentionné dans la question
    question = structure.get("question", "").lower()
    agent = None
    for a in structure.get("agents", []):
        if a.lower() in question:
            agent = a.lower()
            break
    if not agent:
        agent = structure["agents"][0].lower()

    # objet — unique objet extrait
    obj = structure["objects"][0].lower()

    # placement final — lu directement dans tracker.current_facts
    final_location = None
    for fact, value in tracker.current_facts.items():
        if value and fact.startswith(f"{obj}_at_"):
            final_location = fact.split(f"{obj}_at_")[1]
            break

    if not final_location:
        raise ValueError(f"Cannot find final location for {obj} in current_facts")

    fact_str = f"{obj}_at_{final_location}"

    # expected_answer — évalué directement par le reasoner
    r = Reasoner(tracker.model)
    expected = r.evaluate(Formula.knows(agent, Formula.atom(fact_str)))
    r.reset_trace()

    return {
        "query_type"     : "knows_location",
        "agent"          : agent,
        "fact"           : fact_str,
        "expected_answer": expected,
        "reasoning"      : f"Read from Kripke model — no LLM"
    }


def build_tracker_from_structure(structure: dict) -> BeliefTracker:
    agents    = [a.lower() for a in structure["agents"]]
    objects   = [o.lower() for o in structure["objects"]]
    locations = [l.lower() for l in structure["locations"]]

    # normalize events
    events = []
    for ev in structure["events"]:
        normalized = {}
        for k, v in ev.items():
            if isinstance(v, str):
                normalized[k] = v.lower()
            elif isinstance(v, list):
                normalized[k] = [x.lower() if isinstance(x, str) else x for x in v]
            else:
                normalized[k] = v
        events.append(normalized)

    # build initial facts — all false
    initial_facts: dict[str, bool] = {}
    for obj in objects:
        for loc in locations:
            initial_facts[f"{obj}_at_{loc}"] = False

    # set initial location from explicit field
    initial_loc = structure.get("initial_location", "").lower()
    if initial_loc and objects:
        key = f"{objects[0]}_at_{initial_loc}"
        if key in initial_facts:
            initial_facts[key] = True
    else:
        # fallback: first move's from_location
        for ev in events:
            if ev["type"] == "move_object":
                from_loc = ev.get("from_location", "")
                obj      = ev.get("object", "")
                key      = f"{obj}_at_{from_loc}"
                if key in initial_facts:
                    initial_facts[key] = True
                break

    tracker = BeliefTracker(agents=agents, initial_facts=initial_facts)

    for ev in events:
        etype     = ev["type"]
        actor     = ev.get("actor", "")
        witnesses = ev.get("witnesses", [])

        if etype == "agent_leaves":
            tracker.process(Event.leaves(actor, witnesses))

        elif etype == "agent_enters":
            tracker.process(Event.enters(actor, witnesses))

        elif etype == "move_object":
            from_loc = ev.get("from_location", "")
            to_loc   = ev.get("to_location", "")
            obj      = ev.get("object", "")
            if not from_loc or not to_loc:
                continue
            if f"{obj}_at_{from_loc}" not in initial_facts:
                continue
            if f"{obj}_at_{to_loc}" not in initial_facts:
                continue
            tracker.process(Event.move(
                actor=actor,
                witnesses=witnesses,
                obj=obj,
                from_loc=from_loc,
                to_loc=to_loc
            ))

        elif etype == "communicate":
            listener = ev.get("listener") or (witnesses[0] if witnesses else None)
            fact     = ev.get("fact", "")
            value    = ev.get("value", True)
            if listener and fact:
                tracker.process(Event.communicate(
                    speaker=actor,
                    listener=listener.lower() if isinstance(listener, str) else listener,
                    fact=fact,
                    value=value
                ))
        
        elif etype == "agent_looks":
            fact  = ev.get("fact", "")
            value = ev.get("value", True)
            if fact:
                tracker.process(Event.looks(
                    agent=actor,
                    witnesses=witnesses,
                    fact=fact,
                    value=value
                ))

    return tracker


def parse_story(story: str) -> tuple[BeliefTracker, dict, dict]:
    """
    Full pipeline: story text → tracker + question structure.

    Returns:
        tracker   : BeliefTracker with full belief state
        structure : raw extracted JSON
        question  : classified question dict
    """
    structure = extract_story_structure(story)
    tracker   = build_tracker_from_structure(structure)
    question  = build_query_from_tracker(structure, tracker)
    return tracker, structure, question