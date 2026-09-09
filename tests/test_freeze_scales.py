"""--freeze-scales-epoch: from epoch N the Brevitas learned scales stop moving, weights don't."""
import types

import torch

from src.trainer.trainer import Trainer


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixing = torch.nn.Linear(3, 1)
        # mimic Brevitas' parameter naming: <quantizer>.scaling_impl.value
        self.input_quant = torch.nn.Module()
        self.input_quant.scaling_impl = torch.nn.Module()
        self.input_quant.scaling_impl.value = torch.nn.Parameter(torch.tensor(-12.0))


def _trainer(freeze_epoch):
    m = _Model()
    args = types.SimpleNamespace(freeze_scales_epoch=freeze_epoch, num_epoch=8, lr_decay_type='cos',
                                 lr_minibatch=True, save=False, logdir='log/', prefix='x', workdir='./',
                                 predictfile='x', summarize_csv='none', summarize=False, alpha=0)
    t = Trainer.__new__(Trainer)  # skip __init__ (needs dataloaders); only the hook is under test
    t.args, t.model = args, m
    return t, m


def test_scales_freeze_from_given_epoch_only():
    t, m = _trainer(freeze_epoch=6)
    scale = m.input_quant.scaling_impl.value
    for ep in (1, 5):
        t._maybe_freeze_scales(ep)
        assert scale.requires_grad, f"scale must still train at epoch {ep}"
    t._maybe_freeze_scales(6)
    assert not scale.requires_grad
    assert m.mixing.weight.requires_grad, "weights keep training"
    # AdamW must leave the frozen scale untouched (no grad -> skipped, incl. weight decay)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.5)
    before = scale.detach().clone()
    loss = m.mixing(torch.ones(2, 3)).sum()
    loss.backward(); opt.step()
    assert torch.equal(scale.detach(), before)


def test_default_never_freezes():
    t, m = _trainer(freeze_epoch=0)
    for ep in range(1, 9):
        t._maybe_freeze_scales(ep)
    assert m.input_quant.scaling_impl.value.requires_grad
