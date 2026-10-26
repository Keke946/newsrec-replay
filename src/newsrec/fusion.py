"""Label-free fixed-count candidate fusion for offline diagnostics."""


def fixed_count_fusion(base, trend, history, retrieved, slots=10):
    if slots != 10:
        raise ValueError('Only the preregistered ten-slot diagnostic is supported')
    if len(base) != len(set(base)) or len(trend) != len(set(trend)):
        raise ValueError('Duplicate baseline candidates')
    if len(base)>53 or len(trend)>50 or len(retrieved)>10:
        raise ValueError('Candidate budget differs from diagnostic protocol')
    protected = [item for item, _ in reversed(history[-3:]) if item in base]
    priority = protected + trend[:40] + list(retrieved) + list(base)
    picked = list(dict.fromkeys(priority))[:len(base)]
    if len(picked) != len(base) or not set(protected).issubset(picked):
        raise ValueError('Fusion changed count or dropped protected history')
    return picked
