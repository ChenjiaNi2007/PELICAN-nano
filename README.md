# PELICAN Nano Network for Particle Physics

Stripped down version of PELICAN for training ultra-lightweight and interpretable top taggers.
Modified from original PELICAN code: https://arxiv.org/abs/2211.00454

The network is described in arXiv:2310.16121. With a single hidden channel (`--n-hidden 1`)
it has only **11 trainable parameters** (21 nominal, 19 effective). It now supports
**Quantization-Aware Training (QAT)** via [Brevitas](https://github.com/Xilinx/brevitas)
for fixed-point FPGA inference.

---

## Dependencies

```
python >= 3.9
torch >= 1.11
brevitas == 0.12.*   # required only for QAT (--quant)
h5py
scikit-learn
colorlog
pytest
numpy
onnx, onnxscript, onnxoptimizer   # required only for ONNX export
```

Install all dependencies (including QAT support):
```bash
pip install -r requirements.txt
```

---

## General Usage

### Input format

Each datapoint in the HDF5 files contains:
- `Pmu` — per-particle 4-momenta `(E, p_x, p_y, p_z)`
- `is_signal` — binary classification label
- Optionally `truth_Pmu` — true top 4-momentum

The only network input is the Gram matrix of Minkowski dot products `d_ij = p_i · p_j`.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--datadir` | `data/` | Directory containing `train.h5`, `valid.h5`, `test.h5` |
| `--n-hidden` | `1` | Hidden channels (`C_h`). Total params = `10·C_h + 1` (no BN) |
| `--nobj` | `None` | Max particles loaded per event |
| `--nobj-avg` | `49` | Fixed normalisation constant `N̄` used in aggregation |
| `--add-beams` | `True` | Append beam particles `(1,0,0,±1)` to each event |
| `--config` | `s` | Aggregation type for Eq2→2 (`s` = N̄-normalised sums) |
| `--activation` | `relu` | Activation function |
| `--prefix` | `nosave` | Name prefix for all output files |
| `--load` | off | Resume from checkpoint |
| `--cpu` / `--cuda` | auto | Force device |

---

## Training

### Float training (default)

```bash
python train_pelican_nano.py \
    --datadir ./data/sample_data \
    --target is_signal \
    --nobj 80 --nobj-avg 49 \
    --num-epoch 140 \
    --batch-size 256 \
    --prefix my_run \
    --drop-rate 0.05 --drop-rate-out 0.05 \
    --weight-decay 0.005
```

Full toptag dataset command (download from https://osf.io/7u3fk/):

```bash
python train_pelican_nano.py \
    --datadir ./data/toptag \
    --target is_signal \
    --n-hidden 1 --nobj 80 \
    --num-epoch 140 --num-train -1 --num-valid 20000 \
    --batch-size 256 --prefix pelican_nano \
    --drop-rate 0.05 --drop-rate-out 0.05 \
    --weight-decay 0.005
```

### Quantization-Aware Training (QAT)

Add `--quant` to any training command to enable Brevitas QAT. All other arguments remain the same.

```bash
python train_pelican_nano.py \
    --datadir ./data/toptag \
    --target is_signal \
    --n-hidden 1 --nobj 80 \
    --num-epoch 140 --batch-size 256 \
    --prefix pelican_nano_qat \
    --quant
```

#### QAT-specific arguments

| Argument | Default | Description |
|---|---|---|
| `--quant` | off | Enable QAT (Brevitas fake-quantization in the loop) |
| `--weight-bit-width` | `8` | Bit width for weight quantization |
| `--act-bit-width` | `8` | Bit width for activation quantization |
| `--input-bit-width` | `8` | Bit width for `d_ij` input quantization (raise to 16 for better accuracy — inputs are heavy-tailed) |
| `--weight-per-channel` | off | Per-output-channel weight scales instead of per-tensor |
| `--po2-scales` | off | Restrict all scales to powers of two (FPGA-friendly) |
| `--allow-alpha-scaling` | off | Allow `config` chars `S/M/X/N` (N^α scaling) under QAT |

#### Quantization design

- **Input** `d_ij`: quantized by a learned-scale `QuantIdentity` at the network boundary.
- **Post-aggregation**: a `QuantIdentity` is applied after the 6 parameter-free basis ops of Eq2→2 and the 2 ops of Eq2→0, mirroring the FPGA firmware which widens accumulators then rescales by `1/N̄`.
- **Linear mixing**: `QuantLinear` with `Int8WeightPerTensorFloat` weights and float biases (folded at export).
- **Activations**: `QuantReLU` (ReLU) or `QuantIdentity` appended (other activations).
- **Output**: a final `QuantIdentity` on the logit before `cat([-w, w])`.
- Scales are **per-tensor only** — never indexed by particle position, preserving permutation equivariance.

#### Important: checkpoint loading

Brevitas learned scale parameters (`scaling_impl.value`) are initialised during the **first training-mode forward pass**, not at model construction. When loading a QAT checkpoint into a fresh model, a calibration pass is required first:

```python
model.train()
with torch.no_grad():
    model(any_batch)          # initialises quantizer scale params
model.load_state_dict(torch.load('checkpoint.pt')['model_state'], strict=True)
model.eval()
```

The `load_model` function in `scripts/export_qonnx.py` handles this automatically.

---

## ONNX Export

Export a trained model (float or QAT) to ONNX for the hls4ml/FINN downstream flow:

```bash
python scripts/export_qonnx.py \
    --checkpoint model/pelican_nano_qat_best.pt \
    --output model/pelican_nano.onnx \
    --n-hidden 1 \
    --quant-checkpoint          # include if checkpoint is from --quant training
```

Optionally dump integer weights and quantizer scales to a `.pt` file:

```bash
python scripts/export_qonnx.py \
    --checkpoint model/pelican_nano_qat_best.pt \
    --output model/pelican_nano.onnx \
    --quant-checkpoint \
    --dump-weights model/weights.pt
```

The weight dump contains per-layer integer weights, scales, and biases, suitable for feeding directly into the HLS firmware weight loader.

**Note:** the ONNX export uses TorchScript tracing (`dynamo=False`) with dynamic axes for batch size and particle count. `torch.diag_embed` (used in the aggregation ops) is replaced by an equivalent broadcasting expression that is supported by the legacy ONNX exporter.

---

## Checkpoint Conversion (old → new format)

Checkpoints saved before the Phase 1 refactor use `coefs` tensor keys. Convert them for use with the current codebase:

```bash
python scripts/convert_checkpoint.py \
    --input old_checkpoint.pt \
    --output new_checkpoint.pt
```

The FPGA firmware's `model_loader.py` auto-detects both formats and does not require explicit conversion.

---

## Outputs

| Location | Contents |
|---|---|
| `log/` | Per-epoch training/validation stats |
| `model/` | Checkpoints: `<prefix>.pt` (latest), `<prefix>_best.pt` (best validation) |
| `predict/` | Model predictions as `.pt` files |

Re-running with the same `--prefix` overwrites all outputs unless `--load` is used.

---

## Testing

```bash
pytest tests/
```

The test suite covers:
- **Phase 0** — parameter count regression (`10·C_h + 1`), permutation invariance, masking invariance, golden-output non-regression
- **Phase 1** — einsum ↔ Linear mathematical equivalence, checkpoint conversion round-trip
- **Phase 2** — QAT model construction, float-path unchanged, permutation/masking invariance under quantization, `config='S'` guard
- **Phase 3** — CLI arg defaults and wiring, checkpoint save/load with quantizer scales, cross-load failure mode
- **Phase 4** — ONNX export (float and quant), weight dump coverage

---

## Original PELICAN Authors

Alexander Bogatskiy, Flatiron Institute  
Jan T. Offermann, University of Chicago  
Timothy Hoffman, University of Chicago  
Xiaoyang Liu, University of Chicago

---

## Acknowledgments

* [Masked BatchNorm](https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py)
* [Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
* [whichcraft](https://github.com/cookiecutter/whichcraft)

---

## License

This project is licensed under the MIT License — see the LICENSE.md file for details.
