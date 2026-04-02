from llm.bridge import evaluate_story

SALLY_ANNE_STORY = """
Sally and Anne are in the room. Sally puts the ball in the basket.
Sally leaves the room. Anne moves the ball from the basket to the box.
Sally comes back into the room.
Where does Sally think the ball is?
"""

COMMUNICATION_STORY = """
John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Peter enters the kitchen and sees the milk in the cupboard.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?
"""

NESTED_STORY = """
Emma and Liam are in the living room. Emma puts the book on the shelf.
Liam leaves the living room. Emma moves the book from the shelf to the drawer.
Noah enters the living room and sees the book in the drawer.
Emma leaves the living room.
Where does Liam think the book is?
"""

FALSE_BELIEF_TRAP = """
Oliver and Sophia are in the garage. Oliver puts the keys in the toolbox.
Oliver goes to the backyard. Sophia moves the keys from the toolbox to the cabinet.
Oliver comes back to the garage.
Where does Oliver think the keys are?
"""

COMMUNICATION_TRAP = """
Lucas and Mia are in the office. Lucas puts the document in the folder.
Lucas leaves the office. Mia moves the document from the folder to the safe.
Lucas calls Mia on the phone. Mia tells Lucas that the document is in the safe.
Where does Lucas think the document is?
"""

LYING_STORY = """
John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Mary then moves the milk from the cupboard to the drawer.
Peter enters the kitchen and sees the milk in the drawer.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?
"""

if __name__ == "__main__":
    print("\n" + "="*54)
    print("  EpistemicLLM — 3-way comparison")
    print("="*54)

    for name, story in [
        ("Sally-Anne",    SALLY_ANNE_STORY),
        ("Communication", COMMUNICATION_STORY),
        ("Nested",        NESTED_STORY),
        ("False belief trap",  FALSE_BELIEF_TRAP),
        ("Communication trap", COMMUNICATION_TRAP),
        ("Lying story", LYING_STORY),
    ]:
        print(f"\n--- {name} ---")
        result = evaluate_story(story)
        print(result.report())

    print("\nDone.")