import torch
import torch.nn as nn

import logging
from typing import Optional

from .lorentz_metric import dot4
from ..layers import Net2to2, Eq2to0, MessageNet
from ..layers.quant import QuantConfig
from ..trainer import init_weights

try:
    import brevitas.nn as _bnn
    _BREVITAS_AVAILABLE = True
except ImportError:
    _bnn = None
    _BREVITAS_AVAILABLE = False


class PELICANNano(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Aggregator Network
    """
    def __init__(self, n_hidden,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, config='s', config_out='s', average_nobj=49, factorize=False, masked=True,
                 activate_agg_out=True, activate_lin_out=False,
                 scale=1, dropout=True, drop_rate=0.05, drop_rate_out=0.05, batchnorm=None,
                 quant_config: Optional[QuantConfig] = None,
                 n_out: int = 1,
                 head_hidden: int = 0,
                 device=torch.device('cpu'), dtype=None):
        super().__init__()

        logging.info('Initializing network!')

        self.device, self.dtype = device, dtype
        self.n_hidden = n_hidden
        if int(n_out) < 1:
            raise ValueError(f'n_out must be >= 1, got {n_out}')
        self.n_out = int(n_out)
        if int(head_hidden) < 0:
            raise ValueError(f'head_hidden must be >= 0, got {head_hidden}')
        self.head_hidden = int(head_hidden)
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.config = config
        self.config_out = config_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked

        _use_quant = quant_config is not None and quant_config.enabled

        if _use_quant and not _BREVITAS_AVAILABLE:
            raise ImportError('brevitas is required for quantization. Install it with: pip install brevitas')

        # D5: uppercase config chars require explicit opt-in
        if _use_quant and any(c in 'SMXN' for c in (config + config_out)):
            if not quant_config.allow_alpha_scaling:
                raise NotImplementedError(
                    "config contains uppercase chars (N^alpha scaling). "
                    "Pass quant_config.allow_alpha_scaling=True to opt in."
                )

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        # D3: QuantIdentity at the d_ij input boundary (heavy-tailed, needs learned scale)
        if _use_quant:
            from ..layers.quant import make_act_quant
            # input_unsigned drops the sign bit: the Minkowski dot is non-negative,
            # so the whole width goes to magnitude (free bit of dot resolution).
            self.input_quant = _bnn.QuantIdentity(
                act_quant=make_act_quant(quant_config, quant_config.input_bit_width,
                                         unsigned=quant_config.input_unsigned,
                                         clip_min=quant_config.input_clip_min),
                return_quant_tensor=True,
            )
            self.output_quant = _bnn.QuantIdentity(
                act_quant=make_act_quant(quant_config),
                return_quant_tensor=False,
            )
            # Optional QuantIdentity on the raw 4-momenta feeding dot4, so training
            # sees the firmware's input_t momentum grid. None = off: momenta stay
            # float and the model is state-dict-identical to before this field.
            if quant_config.pmu_bit_width is None:
                self.pmu_quant = None
            elif quant_config.pmu_block_fp:
                # Lever 7: per-particle block floating point. Stateless, so this
                # adds no state_dict keys (see src/layers/blockfp.py). With
                # pmu_static_exp (SPS) the exponent is a learned per-slot constant
                # instead, which adds pmu_quant.log2_exp / pmu_quant.exp_initialized.
                from ..layers.blockfp import BlockFPQuant
                self.pmu_quant = BlockFPQuant(
                    quant_config.pmu_bit_width,
                    exp_min=quant_config.pmu_exp_min,
                    exp_max=quant_config.pmu_exp_max,
                    static=quant_config.pmu_static_exp,
                    n_slots=quant_config.pmu_n_slots,
                    floor_batches=quant_config.pmu_exp_floor_batches,
                    fixed=quant_config.pmu_exp_fixed,
                )
            else:
                self.pmu_quant = _bnn.QuantIdentity(
                    act_quant=make_act_quant(quant_config, quant_config.pmu_bit_width),
                    return_quant_tensor=False,
                )
            # --jet-quant-split: separate learned quantizers for the jet-spurion
            # populations of d_ij (and of Pmu). See QuantConfig.jet_quant_split. The
            # spurion slots are FIXED positions (0,1 beams; 2 jet), so per-slot
            # quantizers keep permutation invariance over the constituents (slots 3..).
            # Off: nothing is created and the state dict is identical to before.
            self.jet_quant_split = bool(quant_config.jet_quant_split)
            if self.jet_quant_split:
                if quant_config.pmu_block_fp:
                    raise NotImplementedError(
                        "jet_quant_split with pmu_block_fp: block floating point already "
                        "gives every particle (incl. the jet spurion) its own exponent, so a "
                        "separate jet momentum quantizer is moot there; the d_ij split is "
                        "not wired up for the block-FP path either. Use the uniform pmu grid.")
                # Effective widths: each jet quantizer may override (None = inherit).
                _bw_jet = quant_config.jet_input_bit_width or quant_config.input_bit_width
                _bw_mjet = quant_config.mjet_input_bit_width or quant_config.input_bit_width
                _bw_pjet = quant_config.jet_pmu_bit_width or quant_config.pmu_bit_width
                # d[2,j], d[i,2] (i,j != 2): particle-jet and beam-jet dots
                self.input_quant_jet = _bnn.QuantIdentity(
                    act_quant=make_act_quant(quant_config, _bw_jet,
                                             unsigned=quant_config.input_unsigned,
                                             clip_min=None),
                    return_quant_tensor=False,
                )
                # d[2,2] = m_jet^2 only
                self.input_quant_mjet = _bnn.QuantIdentity(
                    act_quant=make_act_quant(quant_config, _bw_mjet,
                                             unsigned=quant_config.input_unsigned,
                                             clip_min=None),
                    return_quant_tensor=False,
                )
                if quant_config.pmu_bit_width is not None:
                    self.pmu_quant_jet = _bnn.QuantIdentity(
                        act_quant=make_act_quant(quant_config, _bw_pjet),
                        return_quant_tensor=False,
                    )
                else:
                    self.pmu_quant_jet = None
        else:
            self.input_quant = None
            self.output_quant = None
            self.pmu_quant = None
            self.jet_quant_split = False
        if not self.jet_quant_split:
            self.input_quant_jet = None
            self.input_quant_mjet = None
            self.pmu_quant_jet = None
        # Fixed position of the full-jet spurion (collate_fn(add_jet=True) layout:
        # [beam+, beam-, jet, constituents...]). Only used with jet_quant_split; the
        # model cannot see add_jet, so it ASSUMES slot 2 is the jet (the trainer
        # refuses --jet-quant-split without --add-jet).
        self.jet_slot = 2

        # This is the main part of the network -- a sequence of permutation-equivariant 2->2 blocks
        # Each 2->2 block consists of a component-wise messaging layer that mixes channels, followed by the equivariant aggegration over particle indices
        self.net2to2 = Net2to2([1, n_hidden], [[]], activate_agg=activate_agg, activate_lin=activate_lin, activation=activation, dropout=dropout, drop_rate=drop_rate, batchnorm=batchnorm, config=config, average_nobj=average_nobj, factorize=factorize, masked=masked, quant_config=quant_config, device=device, dtype=dtype)

        # The final equivariant block is 2->1 and is defined here manually as a messaging (BatchNorm) layer followed by the 2->0 aggregation layer
        # This messaging layer actually reduces to nothing but a BatchNorm
        self.msg_2to0 = MessageNet([n_hidden], activation=activation, batchnorm=batchnorm, device=device, dtype=dtype)
        # This aggregation layer applies 2 aggregators and mixes them down to C_out = n_out output logits.
        # n_out == 1: a single binary score (positive=predicted signal); n_out == K > 1: K class logits.
        if self.head_hidden == 0:
            self.agg_2to0 = Eq2to0(n_hidden, self.n_out, activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation=activation, config=config_out, factorize=False, average_nobj=average_nobj, quant_config=quant_config, device=device, dtype=dtype)
        else:
            # Nonlinear head (--head-hidden K): 2->0 mixes down to K hidden channels, which
            # pass through the activation (activate_lin=True; QuantReLU under QAT), then a
            # K -> n_out linear layer gives the logits.
            self.agg_2to0 = Eq2to0(n_hidden, self.head_hidden, activate_agg=activate_agg_out, activate_lin=True, activation=activation, config=config_out, factorize=False, average_nobj=average_nobj, quant_config=quant_config, device=device, dtype=dtype)
            if _use_quant:
                from ..layers.quant import make_weight_quant
                self.head = _bnn.QuantLinear(
                    self.head_hidden, self.n_out, bias=True,
                    weight_quant=make_weight_quant(quant_config),
                    bias_quant=None,   # float bias (D6), as in Eq2to0
                    input_quant=None,
                    output_quant=None,
                    return_quant_tensor=False)
                # init_weights only touches plain nn.Linear: give the quant head the same init
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(self.head.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            else:
                self.head = nn.Linear(self.head_hidden, self.n_out, device=device, dtype=dtype)

        self.apply(init_weights)

        logging.info('_________________________\n')
        for n, p in self.named_parameters(): logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))
        logging.info('_________________________\n')

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta = self.prepare_input(data)
        # Snap momenta to the trained grid BEFORE the dots (mirrors firmware input_t);
        # beams are part of Pmu here, so they pass through the same quantizer.
        if self.pmu_quant_jet is not None:
            event_momenta = self._pmu_quant_split(event_momenta)
        elif self.pmu_quant is not None:
            event_momenta = self.pmu_quant(event_momenta)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = dot_products.unsqueeze(-1)

        # D3: quantize d_ij at the network input boundary
        if self.jet_quant_split:
            inputs = self._input_quant_split(inputs)
        elif self.input_quant is not None:
            inputs = self.input_quant(inputs)

        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IRC weighting.
        act1 = self.net2to2(inputs, mask=edge_mask.unsqueeze(-1), nobj=nobj)

        # The last equivariant 2->0 block is constructed here by hand: message layer, dropout, and Eq2to0.
        act2 = self.msg_2to0(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.agg_2to0(act2, nobj=nobj)

        # The output layer applies dropout and an MLP.
        if self.dropout:
            act3 = self.dropout_layer_out(act3)

        # Optional hidden-layer head: K ReLU'd pooled features -> n_out logits.
        if self.head_hidden > 0:
            act3 = self.head(act3)

        # D2: quantize logit at the network output boundary
        if self.output_quant is not None:
            act3 = self.output_quant(act3)

        if self.n_out == 1:
            # Binary head: one logit w -> two-class logits [-w, w] for CrossEntropyLoss.
            prediction = torch.cat([-act3, act3], axis=-1)
        else:
            # K-class head: raw (B, K) logits; CrossEntropyLoss applies the softmax.
            prediction = act3

        if not torch.jit.is_tracing() and torch.isnan(prediction).any():
            logging.info(f"inputs: {torch.isnan(inputs).any()}")
            logging.info(f"act1: {torch.isnan(act1).any()}")
            logging.info(f"act2: {torch.isnan(act2).any()}")
            logging.info(f"prediction: {torch.isnan(prediction).any()}")
            assert False, "There are NaN entries in the output! Evaluation terminated."

        if covariance_test:
            return {'predict': prediction, 'inputs': inputs, 'act1': act1, 'act2': act2, 'act3': act3}
        else:
            return {'predict': prediction}

    def _check_jet_slot(self, n):
        if n <= self.jet_slot:
            raise ValueError(f'jet_quant_split needs the jet spurion at slot {self.jet_slot} '
                             f'(collate with add_jet=True); got only {n} particle slots')

    def _pmu_quant_split(self, pmu):
        """Row jet_slot of Pmu through pmu_quant_jet, every other row through pmu_quant.

        Each quantizer only SEES its own population (the jet row is zeroed before
        pmu_quant, so the 5 TeV jet energy cannot enter its runtime scale stats).
        Quantizers map 0 -> 0, so padded rows stay exactly 0.
        """
        B, N, _ = pmu.shape
        self._check_jet_slot(N)
        is_jet = (torch.arange(N, device=pmu.device) == self.jet_slot).view(1, N, 1)
        q_rest = self.pmu_quant(torch.where(is_jet, torch.zeros_like(pmu), pmu))
        q_jet = self.pmu_quant_jet(pmu[:, self.jet_slot, :])            # (B, 4)
        return torch.where(is_jet, q_jet.unsqueeze(1), q_rest)

    def _input_quant_split(self, inputs):
        """Quantize d_ij with three quantizers, one per population, as a plain tensor.

        inputs: (B, N, N, 1) float dots. Populations (j = jet_slot):
          pair  (i != j and k != j) -> input_quant       (fed with the jet row/col zeroed)
          jet   (exactly one of i,k == j) -> input_quant_jet (fed the row and column, the
                                             (j,j) entry zeroed)
          mjet  (i == k == j)       -> input_quant_mjet  (fed d[j,j] only)
        Each quantizer's runtime statistics therefore come from its own population only.
        The result has no single scale, so it is returned as a plain tensor (downstream
        Net2to2 only uses tensor values; measured bit-identical to feeding a QuantTensor).
        """
        B, N = inputs.shape[0], inputs.shape[1]
        self._check_jet_slot(N)
        j = self.jet_slot
        is_j = torch.arange(N, device=inputs.device) == j                # (N,)
        row_j = is_j.view(1, N, 1, 1)                                    # i == j
        col_j = is_j.view(1, 1, N, 1)                                    # k == j
        off = (~is_j).view(1, N, 1).to(inputs.dtype)

        q_pp = self.input_quant(inputs * (~(row_j | col_j)).to(inputs.dtype))
        q_pp = getattr(q_pp, 'value', q_pp)                              # QuantTensor -> Tensor

        row = inputs[:, j, :, :] * off                                   # d[j,k], (B,N,1)
        col = inputs[:, :, j, :] * off                                   # d[i,j], (B,N,1)
        q_row, q_col = self.input_quant_jet(torch.cat([row, col], dim=1)).split(N, dim=1)
        q_jj = self.input_quant_mjet(inputs[:, j, j, :])                 # (B,1)

        out = torch.where(row_j & ~col_j, q_row.unsqueeze(1), q_pp)
        out = torch.where(col_j & ~row_j, q_col.unsqueeze(2), out)
        out = torch.where(row_j & col_j, q_jj.view(B, 1, 1, 1), out)
        return out

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        particle_ps: :obj:`torch.Tensor`
            4-momenta of the particles
        """
        device, dtype = self.device, self.dtype

        particle_ps = data['Pmu'].to(device, dtype)

        data['Pmu'].requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            scalars = None
        return scalars, particle_mask, edge_mask, particle_ps

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list
