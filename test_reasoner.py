from core.kripke import KripkeModel
from core.reasoner import Reasoner, Formula

def build_sally_anne_model() -> KripkeModel:
    m = KripkeModel()
    m.add_world("w_box",    {"ball_in_box": True,  "ball_in_basket": False})
    m.add_world("w_basket", {"ball_in_box": False, "ball_in_basket": True})
    m.set_actual("w_box")
    m.add_agent("anne")
    m.add_agent("sally")
    m.make_indistinguishable("sally", "w_box", "w_basket")
    return m

def test_basic_knowledge():
    m = build_sally_anne_model()
    r = Reasoner(m)

    # Anne knows ball is in box
    f1 = Formula.knows("anne", Formula.atom("ball_in_box"))
    assert r.evaluate(f1) == True
    print(r.proof_report())
    r.reset_trace()

    # Sally does NOT know ball is in box
    f2 = Formula.knows("sally", Formula.atom("ball_in_box"))
    assert r.evaluate(f2) == False
    print(r.proof_report())
    r.reset_trace()

def test_nested_knowledge():
    m = build_sally_anne_model()
    r = Reasoner(m)

    # Anne knows that Sally does NOT know ball is in box
    # Formula: K_anne(¬K_sally(ball_in_box))
    f = Formula.knows(
            "anne",
            Formula.not_(
                Formula.knows("sally", Formula.atom("ball_in_box"))
            )
        )

    print(f"\nEvaluating: {f}")
    result = r.evaluate(f)
    print(r.proof_report())
    assert result == True, "Anne should know that Sally doesn't know"

def test_belief_possible():
    m = build_sally_anne_model()
    r = Reasoner(m)

    # Sally believes it is possible the ball is in the basket
    f = Formula.believes("sally", Formula.atom("ball_in_basket"))
    result = r.evaluate(f)
    print(f"\nSally believes basket possible: {result}")
    print(r.proof_report())
    assert result == True

if __name__ == "__main__":
    test_basic_knowledge()
    test_nested_knowledge()
    test_belief_possible()
    print("\n✓ All reasoner tests passed.")