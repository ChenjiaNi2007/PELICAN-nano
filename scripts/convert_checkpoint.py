"""
scripts/convert_checkpoint.py

Convert a Phase-0 checkpoint (coefs-based einsum) to a Phase-1 checkpoint
(nn.Linear mixing).  Call once per old checkpoint; new checkpoints can be
loaded directly by the Phase-1 model.

Usage
-----
    python scripts/convert_checkpoint.py \
        --input  model/old_run.pt \
        --output model/old_run_v1.pt

The script is idempotent: running it on an already-converted checkpoint will
raise a clear error (the 'coefs' key won't be present).

Mapping
-------
Eq2to2  (factorize=False):
    old: net2to2.eq_layers.N.coefs  [in_dim, out_dim, 6]
    new: net2to2.eq_layers.N.mixing.weight  [out_dim, in_dim*6]
         (bias and diag_bias are unchanged)

Eq2to0  (factorize=False):
    old: agg_2to0.coefs  [in_dim, out_dim, basis_dim]
         agg_2to0.bias   [1, out_dim]
    new: agg_2to0.mixing.weight  [out_dim, in_dim*basis_dim]
         agg_2to0.mixing.bias    [out_dim]
"""
import argparse
import sys
import torch


def convert_state_dict(old_sd: dict) -> dict:
    """
    Return a new state dict with coefs → mixing.weight (and bias, if applicable).
    """
    new_sd = {}
    converted = []

    for key, val in old_sd.items():
        parts = key.split('.')

        # ---- Eq2to2 coefs -----------------------------------------------
        # pattern: <prefix>.eq_layers.<i>.coefs
        if parts[-1] == 'coefs' and 'eq_layers' in parts:
            # val shape: [in_dim, out_dim, basis_dim]
            in_dim, out_dim, basis_dim = val.shape
            new_weight = val.permute(1, 0, 2).reshape(out_dim, in_dim * basis_dim)
            new_key = key.replace('.coefs', '.mixing.weight')
            new_sd[new_key] = new_weight
            converted.append(f'  {key}  →  {new_key}  {tuple(val.shape)} → {tuple(new_weight.shape)}')
            continue

        # ---- Eq2to0 coefs -----------------------------------------------
        # pattern: agg_2to0.coefs  (or <prefix>.agg_2to0.coefs)
        if parts[-1] == 'coefs' and 'agg_2to0' in parts:
            in_dim, out_dim, basis_dim = val.shape
            new_weight = val.permute(1, 0, 2).reshape(out_dim, in_dim * basis_dim)
            new_key = key.replace('.coefs', '.mixing.weight')
            new_sd[new_key] = new_weight
            converted.append(f'  {key}  →  {new_key}  {tuple(val.shape)} → {tuple(new_weight.shape)}')
            continue

        # ---- Eq2to0 bias ------------------------------------------------
        # Old: shape [1, out_dim].  New: lives in mixing.bias, shape [out_dim].
        if parts[-1] == 'bias' and 'agg_2to0' in parts:
            # Only migrate if shape is [1, out_dim] (old style).
            if val.dim() == 2 and val.shape[0] == 1:
                new_val = val.squeeze(0)
                new_key = key.replace('.bias', '.mixing.bias')
                new_sd[new_key] = new_val
                converted.append(f'  {key}  →  {new_key}  {tuple(val.shape)} → {tuple(new_val.shape)}')
                continue
            # If already 1-D (already converted), leave as-is.

        new_sd[key] = val

    if not converted:
        raise ValueError(
            'No coefs keys found.  The checkpoint may already be converted, '
            'or it does not use factorize=False.'
        )

    print('Converted keys:')
    for line in converted:
        print(line)

    return new_sd


def main():
    parser = argparse.ArgumentParser(description='Convert Phase-0 checkpoint to Phase-1 format')
    parser.add_argument('--input',  required=True, help='Path to old checkpoint (.pt)')
    parser.add_argument('--output', required=True, help='Path to write converted checkpoint (.pt)')
    args = parser.parse_args()

    old_sd = torch.load(args.input, map_location='cpu', weights_only=True)
    new_sd = convert_state_dict(old_sd)
    torch.save(new_sd, args.output)
    print(f'\nSaved converted checkpoint to {args.output}')


if __name__ == '__main__':
    main()
