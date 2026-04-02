from core.kripke import KripkeModel

def test_sally_anne():
    m = KripkeModel()

    # Two possible worlds
    m.add_world("w_box",    {"ball_in_box": True,  "ball_in_basket": False})
    m.add_world("w_basket", {"ball_in_box": False, "ball_in_basket": True})

    # The actual world: ball is in the box (Anne moved it there)
    m.set_actual("w_box")

    # Add agents
    m.add_agent("anne")
    m.add_agent("sally")

    # Anne SAW the move → she can distinguish worlds → no edges added
    # Sally was ABSENT → she cannot distinguish → add edge
    m.make_indistinguishable("sally", "w_box", "w_basket")

    # ── queries ──
    print("\n── Sally-Anne false belief test ──")
    print(m.explain("anne", "ball_in_box"))
    print()
    print(m.explain("sally", "ball_in_box"))

    assert m.knows("anne", "ball_in_box"),        "Anne should know ball is in box"
    assert not m.knows("sally", "ball_in_box"),   "Sally should NOT know"
    assert m.believes_possible("sally", "ball_in_basket"), "Sally thinks basket is possible"

    # Nested: does Anne know that Sally doesn't know?
    anne_knows_sally_doesnt = not m.nested_knows(["anne", "sally"], "ball_in_box")
    print(f"\nDoes Anne know that Sally doesn't know? {anne_knows_sally_doesnt}")
    assert anne_knows_sally_doesnt

    print("\nAll tests passed.")

if __name__ == "__main__":
    test_sally_anne()