"""Binomial interval arithmetic shared by the audit and the independent-null research.

The Wilson score interval is the one used everywhere in this package: it stays inside `[0, 1]`,
does not collapse to a zero-width interval at 0 or 1 successes, and is honest at the small sample
sizes an adjudication budget produces. It lives here rather than in a research module because the
shipped audit reports the same interval on its own claim-tier rows.
"""

import math

WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% score interval for `successes / total`; the full `[0, 1]` range when empty."""
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
