"""
Tests for the SPS static per-slot block-FP momentum exponent
(QuantConfig.pmu_static_exp, BlockFPQuant(static=True)).

Invariants:
  - config validation: static needs pmu_block_fp and pmu_bit_width;
  - state_dict gains pmu_quant.log2_exp / pmu_quant.exp_initialized ONLY when static
    (dynamic block-FP stays stateless -> old checkpoints still strict-load);
  - data init: e_s = clamp(ceil(log2 max|E_s|) - 1, exp_min, exp_max), empty slot -> exp_min;
  - LSQ/STE gradient: clipped-high values push log2_exp up; in-range values follow
    sign(round(q) - q); dy/dx = 1 inside the range;
  - zero in -> zero out exactly (masking), beams (1,0,0,+-1) exact for e in [0, W-2];
  - trains (finite loss, exponent receives gradient), round-trips strict, eval without
    init raises, freeze hook and optimizer group handle it.
"""
import math
import types

import pytest
import torch

from tests.conftest import make_batch

try:
    import brevitas  # noqa: F401
    BREVITAS = True
except ImportError:
    BREVITAS = False

from src.layers.blockfp import BlockFPQuant
from src.layers.quant import QuantConfig
from src.models.pelican_nano import PELICANNano

needs_brevitas = pytest.mark.skipif(not BREVITAS, reason="brevitas not installed")


# ---------------------------------------------------------------- config
def test_config_validation():
    with pytest.raises(ValueError):
        QuantConfig(enabled=True, pmu_bit_width=12, pmu_static_exp=True)          # no block-FP
    with pytest.raises(ValueError):
        QuantConfig(enabled=True, pmu_block_fp=True, pmu_static_exp=True)         # no width
    cfg = QuantConfig(enabled=True, pmu_bit_width=12, pmu_block_fp=True,
                      pmu_static_exp=True, pmu_n_slots=22)
    assert cfg.pmu_static_exp and cfg.pmu_n_slots == 22
    # defaults unchanged
    assert QuantConfig().pmu_static_exp is False


def test_module_rejects_bad_static_config_and_shape():
    with pytest.raises(ValueError):
        BlockFPQuant(12, static=True)                  # n_slots missing
    q = BlockFPQuant(12, static=True, n_slots=4)
    q.train()
    with pytest.raises(ValueError):
        q(torch.ones(2, 5, 4))                         # wrong slot count


# ---------------------------------------------------------------- module
def test_state_only_when_static():
    assert dict(BlockFPQuant(12).state_dict()) == {}
    assert list(BlockFPQuant(12).parameters()) == []
    sd = BlockFPQuant(12, static=True, n_slots=22).state_dict()
    assert set(sd) == {'log2_exp', 'exp_initialized', 'exp_floor', 'floor_batches_seen'}
    assert sd['log2_exp'].shape == (22,)


def test_data_init_values():
    q = BlockFPQuant(12, exp_min=0, exp_max=10, static=True, n_slots=5)
    x = torch.zeros(3, 5, 4)
    x[:, 0] = torch.tensor([1.0, 0, 0, 1.0])            # beam: E=1 -> ceil(0)-1=-1 -> clamp 0
    x[0, 1, 0], x[1, 1, 0] = 100.0, 300.0               # max 300 -> ceil(8.23)-1 = 8
    x[2, 2, 0] = 256.0                                  # exactly po2: ceil(8)-1 = 7 (2^8 covers)
    x[1, 3, 0] = 1e6                                    # clamps to exp_max
    # slot 4 all zero -> exp_min
    q.train()
    q(x)
    assert bool(q.exp_initialized)
    assert q.exponent_table().tolist() == [0, 8, 7, 10, 0]
    assert q.exponent_table().dtype == torch.int64
    # the init happens once only
    q(x * 1000)
    assert q.exponent_table().tolist() == [0, 8, 7, 10, 0]


def test_eval_without_init_raises():
    q = BlockFPQuant(12, static=True, n_slots=3)
    q.eval()
    with pytest.raises(RuntimeError, match="not initialized"):
        q(torch.ones(1, 3, 4))


def _init_static(W, n_slots, e):
    q = BlockFPQuant(W, static=True, n_slots=n_slots)
    with torch.no_grad():
        q.log2_exp.fill_(float(e))
        q.exp_initialized.fill_(True)
    return q


def test_grad_positive_when_everything_clips_high():
    q = _init_static(12, 3, 2)                          # clip 2^3 = 8 GeV
    x = torch.full((4, 3, 4), 50.0, requires_grad=True)
    q.train()
    y = q(x)
    assert torch.allclose(y, torch.full_like(y, (2 ** 11 - 1) * 2.0 ** -10 * 4))
    y.sum().backward()
    assert (q.log2_exp.grad > 0).all()
    # clipped -> no gradient to the input
    assert (x.grad == 0).all()
    # LSQ value: ln2 * 2^e * lsb * q_max per element, 4*4 elements per slot
    expect = math.log(2) * 4 * 2.0 ** -10 * (2 ** 11 - 1) * 16
    assert torch.allclose(q.log2_exp.grad, torch.full((3,), expect), rtol=1e-5)


@pytest.mark.parametrize("frac, sign", [(0.3, -1.0), (0.7, 1.0)])
def test_grad_sign_follows_rounding_residual_inside_range(frac, sign):
    # q = 100 + frac steps: round(q)-q = -frac (frac<.5) or 1-frac (frac>.5).
    # d y/d e = ln2 * 2^e * lsb * (round(q) - q)  -> sign = sign(round(q) - q).
    W, e = 12, 3
    lsb = 2.0 ** -(W - 2)
    q = _init_static(W, 2, e)
    val = (100 + frac) * lsb * 2 ** e
    x = torch.full((2, 2, 4), val, requires_grad=True)
    q.train()
    q(x).sum().backward()
    resid = round(100 + frac) - (100 + frac)
    g = q.log2_exp.grad
    assert (torch.sign(g) == sign).all()
    expect = math.log(2) * 2 ** e * lsb * resid * 8
    assert torch.allclose(g, torch.full((2,), expect), rtol=1e-4)
    # inside the range the input sees a pure straight-through gradient
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_zero_in_zero_out():
    q = _init_static(12, 4, 5)
    x = torch.zeros(3, 4, 4)
    x[0, 1] = torch.tensor([40.0, 10.0, -20.0, 30.0])
    for mode in (q.train, q.eval):
        mode()
        y = q(x)
        assert (y[1:] == 0).all() and (y[0, [0, 2, 3]] == 0).all()


def test_beams_exact_for_e_up_to_W_minus_2():
    W = 12
    beams = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]]])
    for e in range(0, W - 1):                           # e in [0, 10]
        q = _init_static(W, 2, e)
        q.eval()
        assert torch.equal(q(beams), beams), f"beam not exact at e={e}"


def test_integer_exponent_ste_and_runaway_clamp():
    q = _init_static(12, 2, 0)
    with torch.no_grad():
        q.log2_exp.copy_(torch.tensor([3.4, 99.0]))
    q.train()
    q(torch.ones(1, 2, 4))
    assert q.exponent_table().tolist() == [3, 10]
    assert q.log2_exp[1].item() == pytest.approx(10.5)   # clamped to exp_max + 0.5


# ---------------------------------------------------------------- model
def _model(static=True, n_particles=6, seed=0):
    torch.manual_seed(seed)
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                       input_bit_width=6, pmu_bit_width=12, pmu_block_fp=True,
                       pmu_static_exp=static, pmu_n_slots=n_particles + 2,
                       po2_scales=True)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                       dropout=False)


def _batch(n_particles=6, B=8, seed=42):
    b = make_batch(B=B, N_particles=n_particles, seed=seed)
    b['Pmu'] = b['Pmu'].clone()
    b['Pmu'][:, 2:] *= 40.0                             # GeV-ish constituents, beams untouched
    return b


@needs_brevitas
def test_model_state_dict_keys_only_when_static():
    keys_s = set(_model(static=True).state_dict())
    keys_d = set(_model(static=False).state_dict())
    assert {'pmu_quant.log2_exp', 'pmu_quant.exp_initialized'} <= keys_s
    assert not any(k.startswith('pmu_quant.') for k in keys_d)
    assert keys_s - keys_d == {'pmu_quant.log2_exp', 'pmu_quant.exp_initialized',
                               'pmu_quant.exp_floor', 'pmu_quant.floor_batches_seen'}


@needs_brevitas
def test_train_step_smoke_and_optimizer_group():
    from src.trainer.utils import init_optimizer
    model = _model()
    batch = _batch()
    args = types.SimpleNamespace(optim='adamw', lr_init=1e-2, weight_decay=0.005, num_epoch=1)
    opt = init_optimizer(args, model)
    assert len(opt.param_groups) == 2
    g_exp = [g for g in opt.param_groups if any(p is model.pmu_quant.log2_exp for p in g['params'])]
    assert len(g_exp) == 1 and g_exp[0]['weight_decay'] == 0.0 and g_exp[0]['lr'] == 1e-2
    assert len(g_exp[0]['params']) == 1
    n_all = sum(len(g['params']) for g in opt.param_groups)
    assert n_all == len(list(model.parameters()))

    model.train()
    before = None
    for _ in range(2):
        opt.zero_grad()
        out = model(batch)['predict']
        loss = torch.nn.functional.cross_entropy(out, batch['is_signal'])
        assert torch.isfinite(loss)
        loss.backward()
        # the trainer's "missing gradient" warning must not fire for log2_exp
        assert model.pmu_quant.log2_exp.grad is not None
        if before is None:
            before = model.pmu_quant.log2_exp.detach().clone()
        opt.step()
    g = model.pmu_quant.log2_exp.grad
    assert g.abs().sum() > 0 or not torch.equal(before, model.pmu_quant.log2_exp.detach())


@needs_brevitas
def test_single_optimizer_group_without_static():
    from src.trainer.utils import init_optimizer
    model = _model(static=False)
    args = types.SimpleNamespace(optim='adamw', lr_init=1e-2, weight_decay=0.005, num_epoch=1)
    opt = init_optimizer(args, model)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]['weight_decay'] == 0.005


@needs_brevitas
def test_reload_roundtrip_strict():
    model = _model()
    batch = _batch()
    model.train()
    model(batch)
    model.eval()
    with torch.no_grad():
        ref = model(batch)['predict']
    table = model.pmu_quant.exponent_table()

    model2 = _model(seed=1)
    model2.train()
    with torch.no_grad():
        model2(_batch(seed=7))                          # populate Brevitas scale buffers
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.eval()
    with torch.no_grad():
        out = model2(batch)['predict']
    assert torch.equal(ref, out)
    assert torch.equal(table, model2.pmu_quant.exponent_table())


@needs_brevitas
def test_padded_particles_stay_zero_through_model_quantizer():
    model = _model()
    batch = _batch()
    model.train()
    model(batch)
    pmu = batch['Pmu'].clone()
    pmu[:, 5:, :] = 0.0
    assert (model.pmu_quant(pmu)[:, 5:, :] == 0).all()


def test_freeze_scales_freezes_log2_exp():
    from src.trainer.trainer import Trainer

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixing = torch.nn.Linear(3, 1)
            self.pmu_quant = BlockFPQuant(12, static=True, n_slots=4)

    m = _M()
    t = Trainer.__new__(Trainer)
    t.args = types.SimpleNamespace(freeze_scales_epoch=3)
    t.model = m
    t._maybe_freeze_scales(2)
    assert m.pmu_quant.log2_exp.requires_grad
    t._maybe_freeze_scales(3)
    assert not m.pmu_quant.log2_exp.requires_grad
    assert m.mixing.weight.requires_grad


# ---------------------------------------------------------------- exponent floor
def _slot_batch(n_slots, emax_per_slot, B=4):
    """Batch whose slot-s max energy is emax_per_slot[s] (0 -> empty slot)."""
    x = torch.zeros(B, n_slots, 4)
    for s, E in enumerate(emax_per_slot):
        if E > 0:
            x[:, s, 0] = torch.linspace(E / 4, E, B)
            x[:, s, 3] = x[:, s, 0] * 0.5
    return x


def test_floor_buffers_always_present_and_inert_with_K0():
    q = BlockFPQuant(12, static=True, n_slots=22)
    sd = q.state_dict()
    assert set(sd) == {'log2_exp', 'exp_initialized', 'exp_floor', 'floor_batches_seen'}
    assert torch.equal(sd['exp_floor'], torch.zeros(22))
    assert sd['floor_batches_seen'].dtype == torch.int64 and int(sd['floor_batches_seen']) == 0
    q.train()
    q(_slot_batch(22, [1.0] * 2 + [300.0 / (i + 1) for i in range(20)]))
    assert torch.equal(q.exp_floor, torch.zeros(22))          # K=0: floor never moves
    assert int(q.floor_batches_seen) == 0
    tbl = q.exponent_table()
    assert tbl.dtype == torch.int64 and tbl.shape == (22,)


def test_K0_identical_to_pure_formula():
    """K=0 must reproduce e_int = clamp(round(log2_exp), exp_min, exp_max) exactly."""
    torch.manual_seed(3)
    q = _init_static(12, 22, 0)
    with torch.no_grad():
        q.log2_exp.copy_(torch.rand(22) * 12 - 1)             # spans below/above clamp
    ref_e = torch.clamp(torch.round(q.log2_exp.detach().clamp(-0.5, 10.5)), 0, 10)
    x = torch.randn(8, 22, 4) * 30
    q.train()
    y = q(x)
    assert torch.equal(q.exponent_table(), ref_e.to(torch.int64))
    sc = torch.exp2(ref_e)[:, None]
    ref = torch.clamp(torch.round(x / sc / q.mantissa_lsb), q.q_min, q.q_max) * q.mantissa_lsb * sc
    assert torch.equal(y.detach(), ref)


def test_floor_monotone_then_frozen_after_K():
    K = 3
    q = BlockFPQuant(12, static=True, n_slots=4, floor_batches=K)
    q.train()
    seq = [[1.0, 100.0, 5.0, 0.0],      # e: 0, 6, 2, 0(empty)
           [1.0, 20.0, 60.0, 0.0],      # f: 0, 4, 5, 0 -> floor 0,6,5,0
           [1.0, 700.0, 3.0, 9.0],      # f: 0, 9, 1, 3 -> floor 0,9,5,3
           [1.0, 5000.0, 900.0, 900.0]]  # batch 4 > K: frozen
    floors = []
    for i, emax in enumerate(seq):
        q(_slot_batch(4, emax))
        floors.append(q.exp_floor.clone())
        if i > 0:
            assert (floors[i] >= floors[i - 1]).all()
        # effective exponent never below the floor
        assert (q.exponent_table().float() >= q.exp_floor).all()
    assert floors[0].tolist() == [0, 6, 2, 0]
    assert floors[2].tolist() == [0, 9, 5, 3]
    assert torch.equal(floors[3], floors[2])                  # frozen after K batches
    assert int(q.floor_batches_seen) == K
    # first-batch data init and the floor formula agree (K=1 special case)
    q1 = BlockFPQuant(12, static=True, n_slots=4, floor_batches=1)
    q1.train()
    q1(_slot_batch(4, seq[0]))
    assert torch.equal(q1.exp_floor, q1.log2_exp.detach())


def test_effective_exponent_respects_floor_even_if_log2_exp_drifts_down():
    q = BlockFPQuant(12, static=True, n_slots=3, floor_batches=1)
    q.train()
    q(_slot_batch(3, [1.0, 200.0, 40.0]))                     # floor 0, 7, 5
    with torch.no_grad():
        q.log2_exp.fill_(0.0)                                  # optimizer pushes it down
    assert q.exponent_table().tolist() == [0, 7, 5]
    q(_slot_batch(3, [1.0, 2.0, 2.0]))
    assert q.exponent_table().tolist() == [0, 7, 5]
    assert (q.log2_exp.detach() >= q.exp_floor - 0.5).all()  # runaway bound = floor


@needs_brevitas
def test_reload_roundtrip_with_floor():
    def mk(seed):
        torch.manual_seed(seed)
        qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                           input_bit_width=6, pmu_bit_width=12, pmu_block_fp=True,
                           pmu_static_exp=True, pmu_n_slots=8, pmu_exp_floor_batches=4,
                           po2_scales=True)
        return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                           dropout=False)
    model = mk(0)
    model.train()
    for s in range(3):
        model(_batch(seed=s))
    assert int(model.pmu_quant.floor_batches_seen) == 3
    model.eval()
    batch = _batch(seed=99)
    with torch.no_grad():
        ref = model(batch)['predict']
    model2 = mk(1)
    model2.train()
    with torch.no_grad():
        model2(_batch(seed=7))
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.eval()
    with torch.no_grad():
        assert torch.equal(ref, model2(batch)['predict'])
    assert torch.equal(model.pmu_quant.exp_floor, model2.pmu_quant.exp_floor)
    assert int(model2.pmu_quant.floor_batches_seen) == 3


def test_pre_floor_static_state_dict_still_loads_strict():
    """Checkpoints saved before exp_floor existed load as 'no floor' (exp_min)."""
    q = _init_static(12, 3, 4)
    sd = {k: v for k, v in q.state_dict().items() if k in ('log2_exp', 'exp_initialized')}
    q2 = BlockFPQuant(12, static=True, n_slots=3)
    q2.load_state_dict(sd, strict=True)
    assert torch.equal(q2.exp_floor, torch.zeros(3))
    assert torch.equal(q2.exponent_table(), q.exponent_table())


# ---------------------------------------------------------------- fixed (not learned)
def test_fixed_flag_validation():
    with pytest.raises(ValueError):
        QuantConfig(enabled=True, pmu_bit_width=12, pmu_block_fp=True, pmu_exp_fixed=True)
    cfg = QuantConfig(enabled=True, pmu_bit_width=12, pmu_block_fp=True,
                      pmu_static_exp=True, pmu_exp_fixed=True)
    assert cfg.pmu_exp_fixed and QuantConfig().pmu_exp_fixed is False


@pytest.mark.parametrize("K", [0, 3])
def test_fixed_exponent_is_data_derived_and_frozen(K):
    q = BlockFPQuant(12, static=True, n_slots=4, floor_batches=K, fixed=True)
    q.train()
    seq = [[1.0, 100.0, 5.0, 0.0], [1.0, 20.0, 60.0, 0.0],
           [1.0, 700.0, 3.0, 9.0], [1.0, 5000.0, 900.0, 900.0]]
    for emax in seq:
        x = _slot_batch(4, emax).requires_grad_()
        q(x).sum().backward()
        assert not q.log2_exp.requires_grad and q.log2_exp.grad is None
    expect = [0, 6, 2, 0] if K == 0 else [0, 9, 5, 3]
    assert q.exponent_table().tolist() == expect
    assert q.log2_exp.detach().tolist() == [float(v) for v in expect]
    if K:
        assert torch.equal(q.log2_exp.detach(), q.exp_floor)


@needs_brevitas
def test_fixed_train_step_and_reload_roundtrip():
    from src.trainer.utils import init_optimizer

    def mk(seed):
        torch.manual_seed(seed)
        qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6,
                           input_bit_width=6, pmu_bit_width=12, pmu_block_fp=True,
                           pmu_static_exp=True, pmu_n_slots=8, pmu_exp_floor_batches=2,
                           pmu_exp_fixed=True, po2_scales=True)
        return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu',
                           dropout=False)
    model = mk(0)
    args = types.SimpleNamespace(optim='adamw', lr_init=1e-2, weight_decay=0.005, num_epoch=1)
    opt = init_optimizer(args, model)                  # frozen param in the wd=0 group is fine
    model.train()
    for s in range(3):
        opt.zero_grad()
        b = _batch(seed=s)
        loss = torch.nn.functional.cross_entropy(model(b)['predict'], b['is_signal'])
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    table = model.pmu_quant.exponent_table()
    assert torch.equal(model.pmu_quant.log2_exp.detach(), model.pmu_quant.exp_floor)
    model.eval()
    batch = _batch(seed=99)
    with torch.no_grad():
        ref = model(batch)['predict']
    model2 = mk(1)
    model2.train()
    with torch.no_grad():
        model2(_batch(seed=7))
    model2.load_state_dict(model.state_dict(), strict=True)
    model2.eval()
    with torch.no_grad():
        assert torch.equal(ref, model2(batch)['predict'])
    assert torch.equal(table, model2.pmu_quant.exponent_table())
    assert not model2.pmu_quant.log2_exp.requires_grad


# ---------------------------------------------------------------- exponent-only freeze
class _SPSModel(torch.nn.Module):
    """Mimics the real naming: <...>.pmu_quant.log2_exp next to a Brevitas-style scale."""
    def __init__(self):
        super().__init__()
        self.mixing = torch.nn.Linear(4, 1)
        self.pmu_quant = BlockFPQuant(12, static=True, n_slots=4)
        self.input_quant = torch.nn.Module()
        self.input_quant.scaling_impl = torch.nn.Module()
        self.input_quant.scaling_impl.value = torch.nn.Parameter(torch.tensor(-6.0))

    def forward(self, x):
        return self.mixing(self.pmu_quant(x)).sum()


def _exp_trainer(n):
    from src.trainer.trainer import Trainer
    m = _SPSModel()
    t = Trainer.__new__(Trainer)
    t.args = types.SimpleNamespace(pmu_exp_freeze_epoch=n, freeze_scales_epoch=0)
    t.model = m
    return t, m


def test_pmu_exp_freeze_epoch_only_touches_log2_exp():
    t, m = _exp_trainer(3)
    m.train()
    m(_slot_batch(4, [1.0, 100.0, 5.0, 0.0])).backward()
    for ep in (1, 2):
        t._maybe_freeze_pmu_exp(ep)
        assert m.pmu_quant.log2_exp.requires_grad
    t._maybe_freeze_pmu_exp(3)
    assert not m.pmu_quant.log2_exp.requires_grad and m.pmu_quant.log2_exp.grad is None
    assert m.input_quant.scaling_impl.value.requires_grad       # other scales keep training
    assert m.mixing.weight.requires_grad
    t._maybe_freeze_scales(3)                                     # freeze_scales_epoch=0: no-op
    assert m.input_quant.scaling_impl.value.requires_grad


def test_pmu_exp_freeze_epoch1_keeps_data_init_table_frozen():
    t, m = _exp_trainer(1)
    t._maybe_freeze_pmu_exp(1)                                    # before the first batch
    assert not m.pmu_quant.log2_exp.requires_grad
    opt = torch.optim.AdamW(m.parameters(), lr=0.1, weight_decay=0.5)
    m.train()
    for s in range(3):
        opt.zero_grad()
        m(_slot_batch(4, [1.0, 100.0 * (s + 1), 5.0, 0.0])).backward()
        opt.step()
    assert bool(m.pmu_quant.exp_initialized)
    assert m.pmu_quant.exponent_table().tolist() == [0, 6, 2, 0]  # first-batch data init
    assert m.pmu_quant.log2_exp.detach().tolist() == [0.0, 6.0, 2.0, 0.0]
    assert m.pmu_quant.log2_exp.grad is None


# ---------------------------------------------------------------- --init-from warm start
def _qmodel(n_slots=8, static=False, seed=0):
    torch.manual_seed(seed)
    kw = dict(pmu_block_fp=True, pmu_static_exp=True, pmu_exp_floor_batches=2,
              pmu_exp_fixed=True) if static else {}
    qcfg = QuantConfig(enabled=True, weight_bit_width=6, act_bit_width=6, input_bit_width=6,
                       pmu_bit_width=12, pmu_n_slots=n_slots, po2_scales=True,
                       input_clip_min=512, **kw)
    return PELICANNano(2, quant_config=qcfg, batchnorm='b', activation='relu', dropout=False)


def _uniform_ckpt(tmp_path):
    src = _qmodel(static=False, seed=0)
    src.train()
    for s in range(3):
        src(_batch(seed=s))
    path = tmp_path / 'ctl_best.pt'
    torch.save({'model_state': src.state_dict()}, path)
    return src, path


@needs_brevitas
def test_init_from_uniform_into_static_fixed_copies_shared_weights(tmp_path):
    from src.trainer.utils import warm_start_from
    src, path = _uniform_ckpt(tmp_path)
    dst = _qmodel(static=True, seed=5)
    loaded, skipped, fresh = warm_start_from(dst, str(path), _batch(seed=11))
    ssd, dsd = src.state_dict(), dst.state_dict()
    for k in ('net2to2.eq_layers.0.mixing.weight', 'agg_2to0.mixing.weight',
              'net2to2.message_layers.0.normlayer.running_mean', 'msg_2to0.normlayer.weight',
              'input_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value'):
        assert torch.equal(ssd[k], dsd[k]), k
    assert skipped and all(k.startswith('pmu_quant.act_quant.') for k in skipped)
    assert set(fresh) == {'pmu_quant.log2_exp', 'pmu_quant.exp_initialized',
                          'pmu_quant.exp_floor', 'pmu_quant.floor_batches_seen'}
    # the materialization forward must NOT have consumed the SPS data init
    q = dst.pmu_quant
    assert not bool(q.exp_initialized) and int(q.floor_batches_seen) == 0
    assert torch.equal(q.exp_floor, torch.zeros(8))


@needs_brevitas
def test_init_from_then_trains_and_data_init_runs_on_first_batch(tmp_path):
    from src.trainer.utils import warm_start_from, init_optimizer
    _, path = _uniform_ckpt(tmp_path)
    dst = _qmodel(static=True, seed=5)
    warm_start_from(dst, str(path), _batch(seed=11))
    args = types.SimpleNamespace(optim='adamw', lr_init=2.5e-4, weight_decay=0.005, num_epoch=1)
    opt = init_optimizer(args, dst)
    dst.train()
    b = _batch(seed=12)
    w0 = dst.net2to2.eq_layers[0].mixing.weight.detach().clone()
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(dst(b)['predict'], b['is_signal'])
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    q = dst.pmu_quant
    assert bool(q.exp_initialized) and int(q.floor_batches_seen) == 1
    assert torch.equal(q.log2_exp.detach(), q._batch_exponent(b['Pmu']))   # init from THIS batch
    assert not torch.equal(w0, dst.net2to2.eq_layers[0].mixing.weight.detach())


# ---------------------------------------------------------------- loaded Brevitas scales stick
def _runtime_scalers(model):
    from brevitas.core.scaling.standalone import ParameterFromRuntimeStatsScaling
    return {n: m for n, m in model.named_modules() if isinstance(m, ParameterFromRuntimeStatsScaling)}


def _assert_scales_stick(model):
    sc = _runtime_scalers(model)
    assert sc
    before = {n: m.value.detach().clone() for n, m in sc.items()}
    for n, m in sc.items():
        assert m.counter > m.collect_stats_steps, n
    model.train()
    with torch.no_grad():
        for s in range(2):
            model(_batch(seed=40 + s))
    for n, m in sc.items():
        assert torch.equal(m.value.detach(), before[n]), f'{n} value overwritten'
        assert m.counter > m.collect_stats_steps, n


@needs_brevitas
def test_warm_start_scales_are_not_recollected(tmp_path):
    from src.trainer.utils import warm_start_from
    src, path = _uniform_ckpt(tmp_path)
    dst = _qmodel(static=True, seed=5)
    warm_start_from(dst, str(path), _batch(seed=11))
    _assert_scales_stick(dst)
    # and they equal the source's loaded scales
    ssd = src.state_dict()
    for n, m in _runtime_scalers(dst).items():
        assert torch.equal(m.value.detach(), ssd[n + '.value'])


@needs_brevitas
def test_trainer_load_state_scales_are_not_recollected(tmp_path):
    from src.trainer.trainer import Trainer
    src, _ = _uniform_ckpt(tmp_path)
    ck = tmp_path / 'resume.pt'
    torch.save({'model_state': src.state_dict(), 'epoch': 3, 'minibatch': 10,
                'best_metrics': {'loss': 0.3}}, ck)
    t = Trainer.__new__(Trainer)
    t.model = _qmodel(static=False, seed=5)                  # fresh, no forward yet (as --load)
    t.device, t.optimizer, t.scheduler = 'cpu', None, None
    t.args = types.SimpleNamespace(bestfile=str(tmp_path / 'nonexistent.pt'), lr_minibatch=True)
    t.load_state(str(ck))
    _assert_scales_stick(t.model)
