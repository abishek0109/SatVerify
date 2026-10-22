"""Unit tests for Score Fusion and Decision Engine."""

from core.decision_engine import DecisionEngine
from core.score_fusion import WeightedScoreFuser
from forensics.base import ForensicResult


def test_score_fusion_renormalizes_weights_correctly():
    """Verify that when only some branches are available, weights re-normalize to 1.0."""
    fuser = WeightedScoreFuser()

    # Provide only ELA (base wt 0.22) and FFT (base wt 0.22)
    branch_results = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.80),
        ForensicResult(branch_id="fft", branch_name="FFT", status="AVAILABLE", score=0.60),
        ForensicResult(branch_id="vit", branch_name="ViT", status="UNAVAILABLE", score=None),
        ForensicResult(branch_id="structural", branch_name="Structural", status="UNAVAILABLE", score=None),
    ]

    fusion = fuser.fuse(branch_results)
    assert fusion.is_sufficient_evidence is True
    assert fusion.fused_score is not None

    # ELA and FFT had equal base weights (0.22 each), so their effective weights should be 0.50 each
    assert abs(fusion.effective_weights["ela"] - 0.50) < 1e-3
    assert abs(fusion.effective_weights["fft"] - 0.50) < 1e-3
    # Fused score = (0.80 * 0.5) + (0.60 * 0.5) = 0.70
    assert abs(fusion.fused_score - 0.70) < 1e-3


def test_missing_branch_never_treated_as_zero_anomaly():
    """Verify that an unavailable branch is excluded from the math and does not pull score towards 0."""
    fuser = WeightedScoreFuser()

    # High anomaly on the only active branches
    branch_results = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.90),
        ForensicResult(branch_id="fft", branch_name="FFT", status="AVAILABLE", score=0.90),
        ForensicResult(branch_id="vit", branch_name="ViT", status="UNAVAILABLE", score=None),
    ]

    fusion = fuser.fuse(branch_results)
    # If missing branches were treated as 0, the score would drop drastically to ~0.40.
    # Because weights are re-normalized over available branches, the score stays 0.90!
    assert abs(fusion.fused_score - 0.90) < 1e-3


def test_decision_engine_tri_state():
    """Verify REAL, FAKE, and INCONCLUSIVE tri-state classification."""
    fuser = WeightedScoreFuser()
    engine = DecisionEngine(real_threshold=0.38, fake_threshold=0.62)

    # 1. Low score -> REAL
    res_real = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.15),
        ForensicResult(branch_id="fft", branch_name="FFT", status="AVAILABLE", score=0.20),
    ]
    dec_real = engine.evaluate(fuser.fuse(res_real))
    assert dec_real.verdict == "REAL"

    # 2. High score -> FAKE
    res_fake = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.85),
        ForensicResult(branch_id="fft", branch_name="FFT", status="AVAILABLE", score=0.80),
    ]
    dec_fake = engine.evaluate(fuser.fuse(res_fake))
    assert dec_fake.verdict == "FAKE"

    # 3. Mid score in uncertainty interval [0.38, 0.62) -> INCONCLUSIVE
    res_inconcl = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.50),
        ForensicResult(branch_id="fft", branch_name="FFT", status="AVAILABLE", score=0.50),
    ]
    dec_inconcl = engine.evaluate(fuser.fuse(res_inconcl))
    assert dec_inconcl.verdict == "INCONCLUSIVE"


def test_insufficient_evidence_returns_inconclusive():
    """Verify that having fewer than minimum required branches yields INCONCLUSIVE."""
    fuser = WeightedScoreFuser()
    engine = DecisionEngine()

    # Only 1 branch available (min required is 2)
    res_single = [
        ForensicResult(branch_id="ela", branch_name="ELA", status="AVAILABLE", score=0.95),
    ]
    fusion = fuser.fuse(res_single)
    assert fusion.is_sufficient_evidence is False

    decision = engine.evaluate(fusion)
    assert decision.verdict == "INCONCLUSIVE"
    assert "insufficient" in decision.reason.lower()
