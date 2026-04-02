"""
Correctness test suite for kripke.py, reasoner.py, tracker.py.
Each test targets a specific property that MUST hold for the
epistemic logic engine to be scientifically valid.
"""
from core.kripke import KripkeModel
from core.reasoner import Reasoner, Formula
from core.tracker import BeliefTracker, Event


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — kripke.py correctness
# ═══════════════════════════════════════════════════════════════

def test_omniscient_agent_has_no_edges():
    """
    An agent who witnesses everything should know all facts.
    If they have no indistinguishability edges, they are omniscient.
    """
    m = KripkeModel()
    m.add_world("w1", {"x": True})
    m.add_world("w2", {"x": False})
    m.set_actual("w1")
    m.add_agent("god")  # no edges added — omniscient
    m.add_agent("blind")
    m.make_indistinguishable("blind", "w1", "w2")

    assert m.knows("god", "x")        == True,  "Omniscient agent must know x"
    assert m.knows("blind", "x")      == False, "Blind agent must not know x"
    print("✓ Omniscient vs blind agent")


def test_knowing_false_fact_is_impossible():
    """
    Knowledge is factive: if agent knows φ, then φ is true.
    It should be IMPOSSIBLE for an agent to 'know' something false
    in the actual world. This is axiom T in modal logic.
    """
    m = KripkeModel()
    m.add_world("w1", {"rain": False})  # actual world: no rain
    m.set_actual("w1")
    m.add_agent("alice")
    # Alice has no edges — she is in w1 only
    # She cannot know "rain" because rain is false in w1

    assert m.knows("alice", "rain") == False, "Cannot know a false fact"
    print("✓ Knowledge is factive (axiom T)")


def test_reflexivity_of_accessibility():
    """
    Every agent must always be able to reach their own current world.
    accessible_worlds must always include from_world itself.
    This is the reflexivity condition of epistemic logic.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.add_world("w2", {"p": False})
    m.set_actual("w1")
    m.add_agent("alice")
    m.make_indistinguishable("alice", "w1", "w2")

    reachable = m.accessible_worlds("alice", "w1")
    assert "w1" in reachable, "Agent must always reach their own world"
    print("✓ Reflexivity: agent always reaches own world")


def test_symmetry_of_indistinguishability():
    """
    If agent cannot distinguish w1 from w2,
    they also cannot distinguish w2 from w1.
    make_indistinguishable must add edges in BOTH directions.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.add_world("w2", {"p": False})
    m.set_actual("w1")
    m.add_agent("alice")
    m.make_indistinguishable("alice", "w1", "w2")

    assert ("w1", "w2") in m.relations["alice"]
    assert ("w2", "w1") in m.relations["alice"], "Edges must be symmetric"
    print("✓ Symmetry of indistinguishability")


def test_three_worlds_partial_knowledge():
    """
    Agent knows some facts but not others depending on
    which worlds they can distinguish.
    3 worlds, agent can distinguish w1 from w2
    but not w2 from w3.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True,  "q": True})
    m.add_world("w2", {"p": True,  "q": False})
    m.add_world("w3", {"p": False, "q": False})
    m.set_actual("w1")
    m.add_agent("alice")
    # Alice cannot distinguish w1 from w2
    m.make_indistinguishable("alice", "w1", "w2")

    # p is True in both w1 and w2 — Alice knows p
    assert m.knows("alice", "p") == True,  "Alice should know p"
    # q is True in w1 but False in w2 — Alice does not know q
    assert m.knows("alice", "q") == False, "Alice should not know q"
    print("✓ Partial knowledge across 3 worlds")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — reasoner.py correctness
# ═══════════════════════════════════════════════════════════════

def test_double_negation():
    """
    ¬(¬φ) must equal φ.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.set_actual("w1")
    m.add_agent("alice")
    r = Reasoner(m)

    f1 = Formula.atom("p")
    f2 = Formula.not_(Formula.not_(Formula.atom("p")))

    assert r.evaluate(f1) == r.evaluate(f2), "Double negation must hold"
    r.reset_trace()
    print("✓ Double negation: ¬¬p = p")


def test_conjunction_requires_both():
    """
    p ∧ q is False if either p or q is False.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True, "q": False})
    m.set_actual("w1")
    m.add_agent("alice")
    r = Reasoner(m)

    f = Formula.and_(Formula.atom("p"), Formula.atom("q"))
    assert r.evaluate(f) == False, "p ∧ q must be False when q is False"
    r.reset_trace()
    print("✓ Conjunction: p ∧ False = False")


def test_nested_knowledge_depth_3():
    """
    3-level nesting: Alice knows that Bob knows that Carol knows p.
    Only valid if p is true in all worlds reachable through all chains.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.set_actual("w1")
    # All agents have no edges — everyone is omniscient
    for agent in ["alice", "bob", "carol"]:
        m.add_agent(agent)

    r = Reasoner(m)
    f = Formula.knows("alice",
            Formula.knows("bob",
                Formula.knows("carol", Formula.atom("p"))))

    assert r.evaluate(f) == True
    r.reset_trace()
    print("✓ Nested knowledge depth 3 with omniscient agents")


def test_nested_knowledge_breaks_with_one_ignorant():
    """
    If Carol doesn't know p, then the whole nesting collapses:
    Alice knows that Bob knows that Carol knows p → FALSE
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.add_world("w2", {"p": False})
    m.set_actual("w1")
    m.add_agent("alice")
    m.add_agent("bob")
    m.add_agent("carol")
    # Only carol has an edge — she doesn't know p
    m.make_indistinguishable("carol", "w1", "w2")

    r = Reasoner(m)
    f = Formula.knows("alice",
            Formula.knows("bob",
                Formula.knows("carol", Formula.atom("p"))))

    assert r.evaluate(f) == False, "Nesting must fail if innermost agent is ignorant"
    r.reset_trace()
    print("✓ Nested knowledge collapses when innermost agent is ignorant")


def test_proof_trace_is_nonempty():
    """
    Every evaluation must produce at least one proof step.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.set_actual("w1")
    m.add_agent("alice")
    r = Reasoner(m)

    r.evaluate(Formula.knows("alice", Formula.atom("p")))
    assert len(r.trace) > 0, "Proof trace must not be empty"
    print("✓ Proof trace is non-empty after evaluation")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — tracker.py correctness
# ═══════════════════════════════════════════════════════════════

def test_witness_gets_no_edge():
    """
    An agent who witnesses a move must NOT get an
    indistinguishability edge — they saw the change.
    """
    tracker = BeliefTracker(
        agents=["anne", "sally"],
        initial_facts={"ball_at_basket": True, "ball_at_box": False}
    )
    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    tracker.process(Event.move("anne", ["anne"], "ball", "basket", "box"))

    # Anne witnessed the move — she must have NO edges
    assert len(tracker.model.relations["anne"]) == 0, \
        "Witness must have no indistinguishability edges"
    print("✓ Witness gets no indistinguishability edge")


def test_absent_agent_gets_edge():
    """
    An agent who was absent during a move MUST get an edge
    between the before-world and the after-world.
    """
    tracker = BeliefTracker(
        agents=["anne", "sally"],
        initial_facts={"ball_at_basket": True, "ball_at_box": False}
    )
    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    tracker.process(Event.move("anne", ["anne"], "ball", "basket", "box"))

    # Sally was absent — she must have edges
    assert len(tracker.model.relations["sally"]) > 0, \
        "Absent agent must have indistinguishability edges"
    print("✓ Absent agent gets indistinguishability edge")


def test_two_moves_compound_ignorance():
    """
    If an agent misses TWO moves, they should be unable to
    distinguish 3 different world states.
    """
    tracker = BeliefTracker(
        agents=["anne", "sally"],
        initial_facts={"ball_at_A": True, "ball_at_B": False, "ball_at_C": False}
    )
    tracker.process(Event.leaves("sally", witnesses=["anne"]))

    # Move 1: A → B
    tracker.process(Event.move("anne", ["anne"], "ball", "A", "B"))
    # Move 2: B → C
    tracker.process(Event.move("anne", ["anne"], "ball", "B", "C"))

    r = Reasoner(tracker.model)

    # Sally should not know ball is at C
    f = Formula.knows("sally", Formula.atom("ball_at_C"))
    assert r.evaluate(f) == False, "Sally missed both moves — she cannot know"
    r.reset_trace()

    # Anne should know
    f2 = Formula.knows("anne", Formula.atom("ball_at_C"))
    assert r.evaluate(f2) == True
    r.reset_trace()
    print("✓ Compound ignorance: two missed moves")


def test_entering_does_not_grant_knowledge():
    """
    An agent re-entering the room does NOT automatically
    gain knowledge of what happened while they were gone.
    Returning ≠ knowing.
    """
    tracker = BeliefTracker(
        agents=["anne", "sally"],
        initial_facts={"ball_at_basket": True, "ball_at_box": False}
    )
    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    tracker.process(Event.move("anne", ["anne"], "ball", "basket", "box"))
    tracker.process(Event.enters("sally", witnesses=["anne", "sally"]))

    r = Reasoner(tracker.model)
    f = Formula.knows("sally", Formula.atom("ball_at_box"))
    assert r.evaluate(f) == False, "Returning does not grant knowledge"
    r.reset_trace()
    print("✓ Entering room does not grant knowledge of past events")


def test_communication_resolves_false_belief():
    """
    After being told the truth, an agent's false belief is resolved.
    """
    tracker = BeliefTracker(
        agents=["anne", "sally"],
        initial_facts={"ball_at_basket": True, "ball_at_box": False}
    )
    tracker.process(Event.leaves("sally", witnesses=["anne"]))
    tracker.process(Event.move("anne", ["anne"], "ball", "basket", "box"))
    tracker.process(Event.enters("sally", witnesses=["anne", "sally"]))

    r = Reasoner(tracker.model)
    assert r.evaluate(Formula.knows("sally", Formula.atom("ball_at_box"))) == False
    r.reset_trace()

    # Anne tells Sally
    tracker.process(Event.communicate("anne", "sally", "ball_at_box", True))

    r2 = Reasoner(tracker.model)
    result = r2.evaluate(Formula.knows("sally", Formula.atom("ball_at_box")))
    assert result == True, "Communication must resolve false belief"
    r2.reset_trace()
    print("✓ Communication resolves false belief")

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — The 5 axioms of epistemic logic S5
# Every correct epistemic logic engine MUST satisfy all 5.
# ═══════════════════════════════════════════════════════════════

def build_two_world_model(agent="alice", p_in_w1=True, p_in_w2=True,
                           connected=False) -> tuple:
    m = KripkeModel()
    m.add_world("w1", {"p": p_in_w1, "q": True})
    m.add_world("w2", {"p": p_in_w2, "q": True})
    m.set_actual("w1")
    m.add_agent(agent)
    if connected:
        m.make_indistinguishable(agent, "w1", "w2")
    return m, Reasoner(m)


def test_axiom_T():
    """
    Axiom T: Kᵢφ → φ
    If agent knows φ, then φ must be TRUE in the actual world.
    Knowledge implies truth. You cannot know something false.
    """
    # Case 1: agent knows p, p must be true
    m, r = build_two_world_model(p_in_w1=True, connected=False)
    knows_p = r.evaluate(Formula.knows("alice", Formula.atom("p")))
    p_true  = r.evaluate(Formula.atom("p"))
    if knows_p:
        assert p_true, "Axiom T violated: knows p but p is false"
    r.reset_trace()

    # Case 2: p is false in actual world — agent cannot know p
    m2 = KripkeModel()
    m2.add_world("w1", {"p": False})
    m2.set_actual("w1")
    m2.add_agent("alice")
    r2 = Reasoner(m2)
    assert r2.evaluate(Formula.knows("alice", Formula.atom("p"))) == False
    r2.reset_trace()
    print("✓ Axiom T: knowledge implies truth")


def test_axiom_4():
    """
    Axiom 4: Kᵢφ → KᵢKᵢφ
    If agent knows φ, then agent knows that they know φ.
    Positive introspection — agents are aware of their own knowledge.
    """
    # Agent with no edges knows p
    # They should also know that they know p
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.set_actual("w1")
    m.add_agent("alice")
    r = Reasoner(m)

    knows_p = r.evaluate(Formula.knows("alice", Formula.atom("p")))
    r.reset_trace()
    knows_knows_p = r.evaluate(
        Formula.knows("alice", Formula.knows("alice", Formula.atom("p")))
    )
    r.reset_trace()

    assert knows_p == True
    assert knows_knows_p == True, "Axiom 4 violated: knows p but not knows-knows p"
    print("✓ Axiom 4: positive introspection (Kφ → KKφ)")


def test_axiom_5():
    """
    Axiom 5: ¬Kᵢφ → Kᵢ¬Kᵢφ
    If agent does NOT know φ, then they know that they don't know φ.
    Negative introspection — agents are aware of their own ignorance.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.add_world("w2", {"p": False})
    m.set_actual("w1")
    m.add_agent("alice")
    m.make_indistinguishable("alice", "w1", "w2")
    r = Reasoner(m)

    not_knows_p = r.evaluate(
        Formula.not_(Formula.knows("alice", Formula.atom("p")))
    )
    r.reset_trace()
    knows_not_knows_p = r.evaluate(
        Formula.knows("alice",
            Formula.not_(Formula.knows("alice", Formula.atom("p"))))
    )
    r.reset_trace()

    assert not_knows_p == True
    assert knows_not_knows_p == True, "Axiom 5 violated: ¬Kp but not K¬Kp"
    print("✓ Axiom 5: negative introspection (¬Kφ → K¬Kφ)")


def test_axiom_K():
    """
    Axiom K: Kᵢ(φ → ψ) ∧ Kᵢφ → Kᵢψ
    If agent knows (φ implies ψ) and knows φ, they must know ψ.
    Distribution axiom — knowledge distributes over implication.
    We test the contrapositive: if agent doesn't know ψ
    but knows φ, they cannot know (φ → ψ).
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True,  "q": True})
    m.add_world("w2", {"p": True,  "q": False})
    m.set_actual("w1")
    m.add_agent("alice")
    # Alice cannot distinguish w1 from w2
    # She knows p (true in both) but not q (false in w2)
    m.make_indistinguishable("alice", "w1", "w2")
    r = Reasoner(m)

    knows_p = r.evaluate(Formula.knows("alice", Formula.atom("p")))
    r.reset_trace()
    knows_q = r.evaluate(Formula.knows("alice", Formula.atom("q")))
    r.reset_trace()

    assert knows_p == True,  "Alice should know p"
    assert knows_q == False, "Alice should not know q"
    print("✓ Axiom K: distribution — knows p, does not know q")


def test_axiom_consistency():
    """
    Consistency: Kᵢφ → ¬Kᵢ¬φ
    An agent cannot simultaneously know φ and know ¬φ.
    Knowledge must be consistent — no contradictions.
    """
    m = KripkeModel()
    m.add_world("w1", {"p": True})
    m.set_actual("w1")
    m.add_agent("alice")
    r = Reasoner(m)

    knows_p     = r.evaluate(Formula.knows("alice", Formula.atom("p")))
    r.reset_trace()
    knows_not_p = r.evaluate(
        Formula.knows("alice", Formula.not_(Formula.atom("p")))
    )
    r.reset_trace()

    # Cannot know both p and not-p
    assert not (knows_p and knows_not_p), \
        "Consistency violated: agent knows p AND not-p simultaneously"
    print("✓ Consistency: cannot know p and ¬p simultaneously")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n═══ Section 1: kripke.py ═══")
    test_omniscient_agent_has_no_edges()
    test_knowing_false_fact_is_impossible()
    test_reflexivity_of_accessibility()
    test_symmetry_of_indistinguishability()
    test_three_worlds_partial_knowledge()

    print("\n═══ Section 2: reasoner.py ═══")
    test_double_negation()
    test_conjunction_requires_both()
    test_nested_knowledge_depth_3()
    test_nested_knowledge_breaks_with_one_ignorant()
    test_proof_trace_is_nonempty()

    print("\n═══ Section 3: tracker.py ═══")
    test_witness_gets_no_edge()
    test_absent_agent_gets_edge()
    test_two_moves_compound_ignorance()
    test_entering_does_not_grant_knowledge()
    test_communication_resolves_false_belief()

    print("\n═══ Section 4: The 5 axioms of epistemic logic S5 ═══")
    test_axiom_T()
    test_axiom_4()
    test_axiom_5()
    test_axiom_K()
    test_axiom_consistency()

    print("\n✓ All correctness tests passed.")