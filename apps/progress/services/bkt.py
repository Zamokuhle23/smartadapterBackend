"""
Bayesian Knowledge Tracing - the per-skill mastery estimator.

Standard 2-parameter BKT with fixed slip/guess; the posterior after each
attempt becomes the stored mastery estimate for that objective.
"""

# Model parameters (tune on real data later)
P_INIT = 0.30       # prior probability a skill is already known
P_TRANSIT = 0.15    # probability of learning the skill from one practice attempt
P_SLIP = 0.10       # probability a known skill is answered incorrectly
P_GUESS = 0.25      # probability an unknown skill is answered correctly


def posterior_known(p_prior: float, correct: bool) -> float:
    """P(known | evidence)."""
    if correct:
        numerator = p_prior * (1.0 - P_SLIP)
        denominator = numerator + (1.0 - p_prior) * P_GUESS
    else:
        numerator = p_prior * P_SLIP
        denominator = numerator + (1.0 - p_prior) * (1.0 - P_GUESS)
    return min(1.0, max(0.0, numerator / denominator))


def update_mastery(previous_mastery: float | None, correct: bool) -> float:
    """
    Full BKT step: evidence update -> learning transition.
    Returns the new mastery estimate in [0, 1].
    """
    p_prior = P_INIT if previous_mastery is None else previous_mastery
    p_known = posterior_known(p_prior, correct)
    # Learning transition: student may have just learned it from this attempt.
    p_after = p_known + (1.0 - p_known) * P_TRANSIT
    return round(min(1.0, max(0.0, p_after)), 4)
