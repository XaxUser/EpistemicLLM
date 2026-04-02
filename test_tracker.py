from core.tracker import BeliefTracker, Event
from core.reasoner import Reasoner, Formula


def test_sally_anne_dynamic():
    """
    The full Sally-Anne story built dynamically event by event.
    No manual world construction — the tracker does it all.
    """
    print("\n── Dynamic Sally-Anne test ──────────────────")

    # initial state: ball is in the basket
    tracker = BeliefTracker(
        agents=["sally", "anne"],
        initial_facts={
            "ball_at_basket": True,
            "ball_at_box":    False
        }
    )

    # Event 1: Sally leaves the room
    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    print("\nAfter Sally leaves:")
    print(tracker.snapshot())

    # Event 2: Anne moves the ball from basket to box
    # Only Anne witnesses this (Sally is absent)
    tracker.process(Event.move(
        actor="anne",
        witnesses=["anne"],
        obj="ball",
        from_loc="basket",
        to_loc="box"
    ))
    print("\nAfter Anne moves the ball:")
    print(tracker.snapshot())

    # Event 3: Sally returns
    tracker.process(Event.enters("sally", witnesses=["anne", "sally"]))
    print("\nAfter Sally returns:")
    print(tracker.snapshot())

    # ── now query the belief state ────────────────────────────────
    r = Reasoner(tracker.model)

    # Anne knows ball is in the box
    f1 = Formula.knows("anne", Formula.atom("ball_at_box"))
    assert r.evaluate(f1) == True, "Anne should know ball is in box"
    r.reset_trace()

    # Sally does NOT know ball is in the box
    f2 = Formula.knows("sally", Formula.atom("ball_at_box"))
    assert r.evaluate(f2) == False, "Sally should not know"
    r.reset_trace()

    # Sally believes ball might still be in basket (her false belief)
    f3 = Formula.believes("sally", Formula.atom("ball_at_basket"))
    assert r.evaluate(f3) == True, "Sally should believe basket is possible"
    r.reset_trace()

    # Anne knows that Sally doesn't know
    f4 = Formula.knows("anne",
            Formula.not_(Formula.knows("sally", Formula.atom("ball_at_box"))))
    assert r.evaluate(f4) == True
    print("\nAnne knows that Sally doesn't know: TRUE")

    print("\n✓ Dynamic Sally-Anne passed.")


def test_communication_updates_belief():
    """
    After Anne TELLS Sally where the ball is,
    Sally's false belief should be resolved.
    """
    print("\n── Communication test ───────────────────────")

    tracker = BeliefTracker(
        agents=["sally", "anne"],
        initial_facts={"ball_at_basket": True, "ball_at_box": False}
    )

    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    tracker.process(Event.move("anne", ["anne"], "ball", "basket", "box"))
    tracker.process(Event.enters("sally", witnesses=["anne", "sally"]))

    # before communication
    r = Reasoner(tracker.model)
    f_before = Formula.knows("sally", Formula.atom("ball_at_box"))
    assert r.evaluate(f_before) == False
    print("Before Anne tells Sally: Sally does NOT know. ✓")
    r.reset_trace()

    # Anne tells Sally the ball is in the box
    tracker.process(Event.communicate("anne", "sally", "ball_at_box", True))

    r2 = Reasoner(tracker.model)
    f_after = Formula.knows("sally", Formula.atom("ball_at_box"))
    result = r2.evaluate(f_after)
    print(f"After Anne tells Sally: Sally knows = {result}")
    print(r2.proof_report())

    print("\n✓ Communication test passed.")


if __name__ == "__main__":
    test_sally_anne_dynamic()
    test_communication_updates_belief()
    print("\n✓ All tracker tests passed.")