"""
scripts/make_toptag20.py

Build data/toptag20/{train,valid,test}.h5: the full top-tagging dataset
(../train_c.h5, ../valid_c.h5, ../test_c.h5 at the workspace root; 200
constituents/jet, float64, ~10 GB in RAM through src/dataloaders) truncated
to the FIRST --nobj constituents, in the sample_data key layout
(Nobj, Pmu, is_signal, truth_Pmu).

Why: the trainer loads every key of every file fully into RAM
(src/dataloaders/utils.py step 3) and collate_fn then keeps only p[:nobj]
(batch_stack). With --nobj 20 the model never sees constituents 21..200, so
this file is a lossless substitute for training/eval at --nobj 20 and fits
an 8 GB laptop (train Pmu: 1.211M x 20 x 4 x 8 B = 775 MB).

Pmu is stored as float32 by default (--dtype): the model casts momenta to
float32 (`data['Pmu'].to(device, dtype)`, dtype=torch.float) before dot4, so the
float32 file yields bit-identical training inputs at half the RAM (train Pmu
387 MB), which is what lets 3 training processes share an 8 GB laptop.

Nobj is stored RAW (unclipped real-particle count, as in the source file)
so scripts/export_golden.py sees the same value it would from the full file;
the model itself derives multiplicity from the E != 0 mask, not from Nobj.

Usage (repo root, venv python):
  .venv/bin/python scripts/make_toptag20.py --src .. --out data/toptag20 --nobj 20
"""
import argparse, os, time
import h5py
import numpy as np

KEEP = ("Nobj", "is_signal", "truth_Pmu")   # copied verbatim; Pmu is truncated

def convert(src, dst, nobj, block, dtype):
    t0 = time.time()
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        n = f["Pmu"].shape[0]
        for k in KEEP:
            g.create_dataset(k, data=f[k][:])
        d = g.create_dataset("Pmu", shape=(n, nobj, 4), dtype=dtype,
                             chunks=(min(block, n), nobj, 4))
        over = 0
        for s in range(0, n, block):
            e = min(s + block, n)
            d[s:e] = f["Pmu"][s:e, :nobj, :].astype(dtype)
            if nobj < f["Pmu"].shape[1]:
                over += int((f["Pmu"][s:e, nobj, 0] != 0).sum())
            print(f"  {os.path.basename(dst)}: {e}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        g.attrs["source"] = os.path.abspath(src)
        g.attrs["nobj_truncation"] = nobj
        g.attrs["pmu_dtype"] = str(np.dtype(dtype))
        g.attrs["jets_with_more_than_nobj"] = over
    print(f"wrote {dst}: {n} jets, {over} ({100*over/n:.1f}%) had >{nobj} constituents "
          f"(dropped tail), {time.time()-t0:.0f}s", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="..", help="dir holding train_c.h5/valid_c.h5/test_c.h5")
    p.add_argument("--out", default="data/toptag20")
    p.add_argument("--nobj", type=int, default=20)
    p.add_argument("--block", type=int, default=50000)
    p.add_argument("--dtype", default="float32", help="Pmu storage dtype (float32|float64)")
    p.add_argument("--splits", default="valid,test,train")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in a.splits.split(","):
        src = os.path.join(a.src, f"{split}_c.h5")
        dst = os.path.join(a.out, f"{split}.h5")
        if os.path.exists(dst):
            print(f"skip {dst} (exists)"); continue
        convert(src, dst + ".tmp", a.nobj, a.block, np.dtype(a.dtype))
        os.rename(dst + ".tmp", dst)

if __name__ == "__main__":
    main()
