"""Isolated differentiable all-window patient loss,not an adopted training path.

No dataset,checkpoint,endpoint or global-state I/O. No optimizer or GPU dispatch.
Window counts are sampling-support counts,not breast-tissue or lesion weights.
"""
import math
from numbers import Integral
from typing import Sequence, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

WindowKey = Tuple[int, int, int]


def _key(value: Sequence[int]) -> WindowKey:
    if len(value) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in value):
        raise ValueError('Window coordinate must contain three integer indices')
    return tuple(int(v) for v in value)


def patient_log_probabilities(logits: Tensor, starts: Sequence[Sequence[int]],
                              valid_counts: Sequence[int],
                              expected_starts: Sequence[Sequence[int]]) -> Tensor:
    """Compute log arithmetic mean probabilities without detaching gradients.

    This does not select windows or run a predictor. All expected windows must
    already have been visited;empty-support windows remain in that contract.
    The classifier is four-class;the reducer cannot infer or reorder labels.
    """
    if logits.ndim != 2 or logits.shape[1] != 4 or not torch.is_floating_point(logits):
        raise ValueError('Require floating N-by-4 logits')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite window logits')
    keys = [_key(s) for s in starts]
    expected = [_key(s) for s in expected_starts]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Invalid expected window set')
    if len(keys) != logits.shape[0] or len(keys) != len(valid_counts):
        raise ValueError('Window and support cardinalities differ')
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('Missing,duplicated or foreign window')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in valid_counts):
        raise ValueError('Support counts must be nonnegative integers')
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    supported = [i for i in order if valid_counts[i] > 0]
    if not supported:
        raise ValueError('No acquired support;no patient output')
    # Promoting low-precision logits is an explicit loss-reduction policy,
    # not a declaration that the original network forward ran in FP32.
    computation = logits if logits.dtype == torch.float64 else logits.float()
    logp = F.log_softmax(computation[supported], dim=1).double()
    return torch.logsumexp(logp, dim=0) - math.log(len(supported))


def patient_loss_terms(log_probabilities: Tensor, target: int,
                       class_weights: Tensor) -> Tuple[Tensor, Tensor]:
    """Return numerator/denominator for corrected weighted-mean accumulation.

    Class weights must be supplied from the fitting partition by a separately
    validated runner. This primitive never estimates weights from outcomes.
    """
    if log_probabilities.shape != (4,) or not torch.is_floating_point(log_probabilities):
        raise ValueError('Require one four-class patient log-probability vector')
    if not bool(torch.isfinite(log_probabilities).all()):
        raise ValueError('Invalid patient log probabilities')
    if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= target < 4:
        raise ValueError('Target must be an integer class index')
    if class_weights.shape != (4,) or not torch.is_floating_point(class_weights):
        raise ValueError('Require four floating class weights')
    if class_weights.requires_grad or not bool(torch.isfinite(class_weights).all()) or bool((class_weights <= 0).any()):
        raise ValueError('Weights must be finite,positive and fixed')
    if class_weights.device != log_probabilities.device:
        raise ValueError('Weights and patient output must share a device')
    if abs(float(torch.logsumexp(log_probabilities.detach().double(), dim=0))) > 1e-6:
        raise ValueError('Patient probabilities do not sum to one')
    denominator = class_weights[int(target)].to(dtype=log_probabilities.dtype)
    return -log_probabilities[int(target)] * denominator, denominator
