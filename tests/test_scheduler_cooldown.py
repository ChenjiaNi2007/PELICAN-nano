"""
Regression test for src/trainer/scheduler.GradualCooldownScheduler.

Builds the exact scheduler stack Trainer builds for --lr-decay-type cos
(CosineAnnealingLR -> GradualWarmupScheduler(4 epochs) -> GradualCooldownScheduler(3 epochs))
with per-minibatch stepping and checks that the cooldown phase decays the LR
geometrically to lr_final over its window instead of halving it every step
(the pre-fix behaviour: LR ~1e-40 after one cooldown epoch, later epochs dead).
"""
import math

import torch
import torch.optim.lr_scheduler as sched

from src.trainer.scheduler import GradualWarmupScheduler, GradualCooldownScheduler


def _build(num_epoch, steps_per_epoch, lr_init=2.5e-3, lr_final=1e-6, warmup_epochs=4, cooldown_epochs=3):
    """Mirror Trainer.__init__ + init_scheduler for lr_decay_type='cos', lr_minibatch=True."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=lr_init)
    base = sched.CosineAnnealingLR(opt, num_epoch * steps_per_epoch, eta_min=lr_final)
    warm = GradualWarmupScheduler(opt, multiplier=1, warmup_epochs=steps_per_epoch * warmup_epochs, after_scheduler=base)
    cooldown_start = (num_epoch - warmup_epochs - cooldown_epochs) * steps_per_epoch
    cooldown_len = cooldown_epochs * steps_per_epoch
    full = GradualCooldownScheduler(opt, lr_final, cooldown_start, cooldown_len, warm)
    return opt, full


def _trace(num_epoch, steps_per_epoch, **kw):
    opt, s = _build(num_epoch, steps_per_epoch, **kw)
    lrs = []
    for _ in range(num_epoch * steps_per_epoch):
        opt.step()
        s.step()
        lrs.append(s.get_last_lr()[0])
    return lrs


def test_cooldown_reaches_lr_final_not_zero():
    E, spe, lr_init, lr_final = 8, 100, 2.5e-3, 1e-6
    lrs = _trace(E, spe, lr_init=lr_init, lr_final=lr_final)
    # end of warmup: at the peak
    assert math.isclose(lrs[4 * spe - 1], lr_init, rel_tol=0.02)
    # cooldown (last 3 epochs) is a geometric decay ending at lr_final, never far below it
    cool = lrs[(E - 3) * spe:]
    assert all(b <= a * 1.0001 for a, b in zip(cool, cool[1:])), "cooldown LR must be non-increasing"
    assert cool[-1] >= lr_final * 0.99, f"final LR {cool[-1]:.3e} undershoots lr_final"
    assert cool[-1] <= lr_final * 1.5, f"final LR {cool[-1]:.3e} did not reach lr_final"
    # one cooldown epoch in, the LR must still be alive (pre-fix it was ~1e-40 here)
    one_epoch_in = cool[spe - 1]
    assert one_epoch_in > 1e-5, f"LR one cooldown epoch in = {one_epoch_in:.3e} (dead)"
    # and the per-step decay ratio is the geometric one, not 0.5
    ratio = cool[10] / cool[9]
    expected = (lr_final / cool[0]) ** (1.0 / (3 * spe))
    assert math.isclose(ratio, expected, rel_tol=0.05), (ratio, expected)


def test_pre_cooldown_untouched_and_longer_runs():
    E, spe = 20, 50
    lrs = _trace(E, spe)
    # cosine phase (epochs 5..17) is strictly below the peak and decreasing
    cos_phase = lrs[4 * spe: (E - 3) * spe]
    assert all(b <= a * 1.0001 for a, b in zip(cos_phase, cos_phase[1:]))
    assert cos_phase[0] <= 2.5e-3 * 1.0001
    # cooldown start continues from the cosine LR (no jump)
    assert math.isclose(lrs[(E - 3) * spe], lrs[(E - 3) * spe - 1], rel_tol=0.05)
    assert lrs[-1] >= 1e-6 * 0.99 and lrs[-1] <= 1e-6 * 1.5
