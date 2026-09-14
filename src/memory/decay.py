import math
from typing import List, Tuple, Dict, Any


def compute_exponential_decay(elapsed_days: float, half_life_days: float = 7.0) -> float:
    """
    Computes time decay using exponential half-life formula:
    decay(t) = 2 ^ (-t / H)
    where t is elapsed_days and H is half_life_days (default: 7.0).
    """
    if elapsed_days < 0:
        elapsed_days = 0.0
    return math.pow(2.0, -elapsed_days / half_life_days)


def aggregate_decayed_signals(signals_with_times: List[Tuple[float, float]], current_time: float, half_life_days: float = 7.0) -> float:
    """
    Aggregates a list of (signal_weight, timestamp) tuples with exponential decay:
    aggregated = sum(signal_i * decay(t_i)) / sum(decay(t_i))
    where t_i is (current_time - timestamp_i) in days.
    """
    if not signals_with_times:
        return 0.0

    weighted_sum = 0.0
    decay_sum = 0.0

    for signal, ts in signals_with_times:
        elapsed_seconds = max(0.0, current_time - ts)
        elapsed_days = elapsed_seconds / 86400.0
        decay = compute_exponential_decay(elapsed_days, half_life_days)
        weighted_sum += signal * decay
        decay_sum += decay

    if decay_sum <= 0.0:
        return 0.0

    aggregated = weighted_sum / decay_sum
    return max(0.0, min(1.0, aggregated))

