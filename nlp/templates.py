STORY_EXTRACTION_PROMPT = """
You are a precise story analyst for a Theory of Mind epistemic logic system.

THEORY OF MIND STORY STRUCTURE:
These stories always follow this pattern:
  1. An object is placed at an initial location (starting state — NOT a move event)
  2. Agents enter or leave the scene
  3. The object gets MOVED from one location to another while some agents are absent
  4. Optionally: an agent explicitly SEES the object at its current location
  5. Optionally: an agent communicates the object's location to another agent
  6. A question asks what an agent believes about the object's location

CRITICAL DEFINITIONS:
- agents: people/characters only (lowercase names)
- objects: the single physical item that gets moved during the story
  RULE: containers (basket, box, fridge, cupboard, bag) are NEVER objects
- locations: the specific places where the object is stored during the story
  RULE: include ALL locations the object occupies during the story
  RULE: "room", "kitchen", "garden", "school" are SCENES not locations — exclude them
- initial_location: where the object is at the start of the story (before any move)

EVENT TYPES — use exactly these 5 types:
  1. agent_leaves   : an agent exits the scene
  2. agent_enters   : an agent returns to the scene (does NOT grant knowledge automatically)
  3. move_object    : the object is physically moved from one location to another
  4. agent_looks    : an agent EXPLICITLY sees/finds/notices the object at its location
                      Use when story says: "sees", "notices", "finds", "looks at"
                      IMPORTANT: agent_enters does NOT imply agent_looks
                      Only use agent_looks when explicitly stated in the story
  5. communicate    : an agent tells another agent where the object is

EVENT RULES:
1. Do NOT create move_object for the initial placement — use initial_location instead
2. Only create move_object for REAL moves AFTER the initial placement
3. witnesses for agent_leaves: agents present BEFORE the actor leaves. Actor NOT included.
4. witnesses for agent_enters: the actor AND all agents already present
5. witnesses for move_object: ONLY agents physically present at that exact moment
6. witnesses for agent_looks: the actor AND agents present at that moment
7. For communicate: always include listener, fact (object_at_location), value (true/false)

Return ONLY valid JSON:
{{
  "agents": ["<agent1>", "<agent2>"],
  "objects": ["<object_that_moves>"],
  "locations": ["<location1>", "<location2>"],
  "initial_location": "<location_where_object_starts>",
  "events": [
    {{
      "type": "agent_leaves",
      "actor": "<who_leaves>",
      "witnesses": ["<agents_still_present>"]
    }},
    {{
      "type": "move_object",
      "actor": "<who_moves_it>",
      "witnesses": ["<agents_present>"],
      "object": "<object_name>",
      "from_location": "<current_location>",
      "to_location": "<new_location>"
    }},
    {{
      "type": "agent_enters",
      "actor": "<who_enters>",
      "witnesses": ["<who_enters>", "<agents_already_present>"]
    }},
    {{
      "type": "agent_looks",
      "actor": "<who_sees>",
      "witnesses": ["<who_sees>", "<agents_present>"],
      "fact": "<object>_at_<location>",
      "value": true
    }},
    {{
      "type": "communicate",
      "actor": "<speaker>",
      "listener": "<listener>",
      "witnesses": ["<speaker>", "<listener>"],
      "fact": "<object>_at_<location>",
      "value": true
    }}
  ],
  "question": "<copy the last sentence of the story exactly>",
  "answer": "<location where the queried agent last saw the object>"
}}

No markdown, no explanation, only JSON.

Story:
{story}
"""