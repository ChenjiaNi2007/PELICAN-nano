"""Resuming must reproduce the LR schedule by replay (not by unpickling the wrapper stack)."""
import torch
import torch.optim.lr_scheduler as sched

from src.trainer.scheduler import GradualWarmupScheduler, GradualCooldownScheduler


def _stack(opt, E=20, spe=50, lr_final=1e-6):
    base = sched.CosineAnnealingLR(opt, E * spe, eta_min=lr_final)
    warm = GradualWarmupScheduler(opt, multiplier=1, warmup_epochs=4 * spe, after_scheduler=base)
    return GradualCooldownScheduler(opt, lr_final, (E - 7) * spe, 3 * spe, warm)


def test_replay_matches_uninterrupted_and_loaded_state_drives_live_optimizer():
    E, spe = 20, 50
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=2.5e-3)
    s = _stack(opt, E, spe)
    ref = []
    for _ in range(E * spe):
        opt.step(); s.step(); ref.append(opt.param_groups[0]["lr"])
    k = 9 * spe  # "crash" after 9 epochs
    # resume by replay on a fresh stack
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt2 = torch.optim.AdamW([p2], lr=2.5e-3)
    s2 = _stack(opt2, E, spe)
    for _ in range(k):
        s2.step()
    out = []
    for _ in range(E * spe - k):
        opt2.step(); s2.step(); out.append(opt2.param_groups[0]["lr"])
    assert all(abs(a - b) <= 1e-12 + 1e-9 * a for a, b in zip(ref[k:], out)), "replayed LR must equal the uninterrupted schedule"
    # and the LR keeps changing after the resume point (the live optimizer is driven)
    assert out[0] != out[spe], "LR must keep evolving after resume"
