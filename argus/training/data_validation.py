"""Binary-label validation and worker-safe extraction failure reporting."""
import warnings

import torch
from torch.utils.data import default_collate


def count_binary_labels(labels):
    values = torch.as_tensor(labels)
    if values.ndim != 1 or not torch.isfinite(values).all() or not ((values == 0) | (values == 1)).all():
        raise ValueError('Expected a one-dimensional vector of finite binary labels (0 or 1)')
    return {0: int((values == 0).sum()), 1: int((values == 1).sum())}


def validate_label_counts(counts, split, require_both=False):
    if sum(counts.values()) == 0:
        raise ValueError(f'{split}: empty split; binary metrics are invalid')
    if not all(counts.values()):
        message = f'{split}: single-class split, label_counts={counts}; binary metrics are invalid'
        if require_both:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


def collate_samples(items):
    """Carry failures from DataLoader workers to the parent; never zero-fill them.

    The trainer rejects any batch containing failures before optimization/metrics.
    Counts cover the attempted batch, not unvisited samples after fail-fast abort.
    """
    failures = [item for item in items if 'error_stage' in item]
    valid = [item for item in items if 'error_stage' not in item]
    batch = default_collate(valid) if valid else {}
    batch['failures'] = failures
    return batch
