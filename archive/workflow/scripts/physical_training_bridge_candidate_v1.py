"""Trainable complete-window bridge; candidate only, not a release permission.

Pure tensor operations with caller-supplied windows. No labels, data files,
checkpoints, GPU allocation or optimizer are opened at import. CPU fixtures
may exercise this module; patient execution needs a separate stage contract.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from physical_patient_loss_candidate_v1 import patient_log_probabilities, patient_loss_terms, _key

Key = Tuple[int, int, int]


@dataclass(frozen=True)
class PatientWindows:
    pid: str
    expected_starts: Sequence[Sequence[int]]
    # Return C,S,A,R image and S,A,R acquisition support for one coordinate.
    get_window: Callable[[Key], Tuple[Tensor, Tensor]]


def _validate_plan(patient: PatientWindows) -> List[Key]:
    keys = [_key(k) for k in patient.expected_starts]
    if not patient.pid or not keys or len(keys) != len(set(keys)):
        raise ValueError('Unique nonempty patient and complete coordinate plan required')
    if len(keys) > 4096 or any(min(key) < 0 for key in keys):
        raise ValueError('Invalid or excessive window plan')
    return sorted(keys)


def patient_forward(net: nn.Module, patient: PatientWindows, device: torch.device,
                    window_microbatch: int = 4, precision: str = 'fp32',
                    checkpoint_activations: bool = True,
                    augmentation: Tuple[bool, bool, float] = (False, False, 1.0)) -> Tuple[Tensor, dict]:
    """Visit every coordinate once and keep differentiable patient probabilities.

    Activation checkpointing bounds retained feature maps, not source-window
    count. RNG is preserved for dropout recomputation. Models with mutable
    buffers are refused rather than silently applying running updates twice.
    Selection uses net.eval/no_grad and disables checkpointing automatically.
    """
    if window_microbatch != 4:
        raise ValueError('Fixed window microbatch4 required')
    if precision not in ('fp32', 'bf16') or (precision == 'bf16' and device.type != 'cuda'):
        raise ValueError('BF16 candidate is CUDA-only;CPU tests use explicit FP32')
    if len(augmentation) != 3 or not all(isinstance(v, bool) for v in augmentation[:2]):
        raise ValueError('Paired A/R flips and one shared gain required')
    flip_a, flip_r, gain = augmentation
    if isinstance(gain, bool) or not .95 <= float(gain) <= 1.05:
        raise ValueError('Gain outside fixed range')
    if checkpoint_activations and net.training and list(net.named_buffers()):
        raise ValueError('Checkpoint candidate requires no mutable model buffers')
    keys = _validate_plan(patient)
    logits_parts, counts, visited = [], [], []
    shape = None
    for offset in range(0, len(keys), window_microbatch):
        batch = []
        for key in keys[offset:offset+window_microbatch]:
            image, valid = patient.get_window(key)
            if image.device.type != 'cpu' or valid.device.type != 'cpu':
                raise ValueError('CPU source windows required before explicit transfer')
            if image.ndim != 4 or image.shape[0] != 3 or image.dtype != torch.float32:
                raise ValueError('Require FP32 phase,S,A,R with three channels')
            if valid.dtype != torch.bool or tuple(valid.shape) != tuple(image.shape[1:]):
                raise ValueError('Explicit matching boolean acquisition support required')
            if shape is None:
                shape = tuple(image.shape)
            if tuple(image.shape) != shape or not bool(torch.isfinite(image).all()):
                raise ValueError('Inconsistent or nonfinite windows')
            if bool((image[:, ~valid] != 0).any()):
                raise ValueError('Unsupported input must remain zero')
            if bool((image < 0).any()) or bool((image > 5).any()):
                raise ValueError('Expected already normalized0..5;no hidden rescaling')
            if flip_a:
                image = image.flip(2)
            if flip_r:
                image = image.flip(3)
            image = (image * float(gain)).clamp(0, 5)
            batch.append(image)
            counts.append(int(valid.sum()))
            visited.append(key)
        xb = torch.stack(batch).to(device=device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
        with context:
            if checkpoint_activations and net.training and torch.is_grad_enabled():
                logits = checkpoint(net, xb, use_reentrant=False, preserve_rng_state=True)
            else:
                logits = net(xb)
        if logits.shape != (len(batch), 4):
            raise ValueError('Classifier must produce four logits per visited window')
        logits_parts.append(logits)
    logp = patient_log_probabilities(torch.cat(logits_parts), visited, counts, keys)
    return logp, dict(pid=patient.pid, windows_visited=len(visited),
                      supported_windows=sum(n > 0 for n in counts),
                      zero_support_windows=sum(n == 0 for n in counts),
                      all_expected_once=True, canonical_coordinate_order=True,
                      augmentation_shared_across_patient_windows=True)


def optimize_patient_group(net: nn.Module, optimizer: torch.optim.Optimizer,
                           patients: Sequence[PatientWindows], targets: Sequence[int],
                           class_weights: Tensor, device: torch.device,
                           augmentations: Sequence[Tuple[bool, bool, float]],
                           precision: str = 'fp32', checkpoint_activations: bool = True) -> dict:
    """One effective group of at most16patients,not16windows or32GPU ranks.

    Normalize by target-weight mass for the whole group and backward per patient.
    A failure clears gradients and performs no optimizer step for this group.
    Production is separately fixed toAdamW;CPU fixtures may use a tiny optimizer.
    """
    if not 1 <= len(patients) <= 16 or len(targets) != len(patients) or len(augmentations) != len(patients):
        raise ValueError('One nonempty effective group of at most16patients required')
    if len({p.pid for p in patients}) != len(patients):
        raise ValueError('Duplicate patient in effective group')
    if class_weights.shape != (4,) or class_weights.requires_grad or not torch.is_floating_point(class_weights):
        raise ValueError('Fixed four-class fitting weights required')
    if class_weights.device != device or not bool(torch.isfinite(class_weights).all()) or bool((class_weights <= 0).any()):
        raise ValueError('Finite positive weights on the training device required')
    if any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < 4 for y in targets):
        raise ValueError('Fixed class-index targets required')
    model_parameters = {id(p) for p in net.parameters() if p.requires_grad}
    optimizer_parameters = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(optimizer_parameters) != model_parameters or len(set(optimizer_parameters)) != len(optimizer_parameters):
        raise ValueError('Optimizer must own exactly this model trainable parameters')
    denominator = class_weights[list(targets)].double().sum().detach()
    optimizer.zero_grad(set_to_none=True)
    records, numerators = [], []
    net.train()
    try:
        for patient, target, aug in zip(patients, targets, augmentations):
            logp, receipt = patient_forward(net, patient, device, precision=precision,
                checkpoint_activations=checkpoint_activations, augmentation=aug)
            numerator, _ = patient_loss_terms(logp, target, class_weights)
            loss = numerator / denominator
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite patient-group objective')
            loss.backward()
            numerators.append(float(numerator.detach()))
            records.append(receipt)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return dict(effective_patients=len(patients), target_weight_mass=float(denominator),
                weighted_patient_loss=sum(numerators)/float(denominator),
                preclip_gradient_norm=float(norm), optimizer_steps=1, patients=records,
                synthetic_or_patient_scope_determined_by_caller=True)


def fit_class_weights(targets: Sequence[int]) -> Tensor:
    """Compute inverse frequency using only the supplied fitting targets."""
    if not targets or any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < 4 for y in targets):
        raise ValueError('Fitting-only integer targets required')
    counts = torch.bincount(torch.tensor(targets), minlength=4)
    if bool((counts == 0).any()):
        raise ValueError('All four fitting classes required')
    return (len(targets)/(4*counts.double())).float()
