"""
scripts/make_hls4ml5.py

Build data/<name>/{train,valid,test}.h5 from the hls4ml LHC 5-class jet dataset
(raw files at the workspace root: "../train 2/*.h5" (62 files, 620k jets) and
"../val 2/*.h5" (26 files, 260k jets)), in the key layout the trainer loads
(src/dataloaders: every key of every file is read fully into RAM, so ONLY
Pmu / Pjet / Nobj / label / is_signal are written).

Raw layout (per file, 10000 jets):
  jetConstituentList (n, 150, 16) float64, columns
    [px, py, pz, E, Erel, pT, pTrel, eta, etarel, etarot, phi, phirel,
     phirot, deltaR, costheta, costhetarel]   (particleFeatureNames has a 17th
    name, j1_pdgid, with no column), pT-sorted, zero-padded at the end (E == 0).
  jets (n, 59) float64, last 6 cols one-hot [j_g, j_q, j_w, j_z, j_t, j_undef].
  jetImage* datasets are never read.

Per jet:
  Pmu       = constituents (E, px, py, pz) [raw cols 3,0,1,2] -- collate_fn order.
  --pt-min  : particles with pT <= pt_min are zeroed (all 4 components), as in
              l1-jet-id fast_jetclass/data/data.py::_cut_transverse_momentum.
  then a stable argsort by pT descending (no-op on raw order; packs survivors to
  the front after a cut), truncate/zero-pad to --nobj.
  Pjet      = (4,) float (E,px,py,pz) FULL-jet 4-momentum: sum over ALL constituents
              AFTER the --pt-min cut, BEFORE truncation (the leading-N Pmu keeps only
              part of the jet; Pjet carries the rest -- used by the trainer's --add-jet
              third spurion). Stored with the --dtype of Pmu.
  Nobj      = int16 count of E != 0 AFTER the cut, BEFORE truncation (raw,
              unclipped, as make_toptag20 stores it; the model masks on E != 0).
  label     = int8 argmax of one-hot g=0, q=1, w=2, z=3, t=4.
  is_signal = int8 (label == 4): top vs rest, keeps the binary tooling usable.

Splits (deterministic, order-preserving, no RNG):
  test.h5  = all jets of --src-val in sorted-file order.
  valid.h5 = --src-train jets whose GLOBAL running index (sorted-file order,
             counted across files) % valid_every == 0; train.h5 = the rest.
  With 620k train jets and valid_every=10: 62,000 valid / 558,000 train.

Files are processed one at a time (one raw file ~190 MB) into pre-sized
chunked datasets; written to <file>.tmp then renamed.

Usage (repo root, venv python):
  .venv/bin/python scripts/make_hls4ml5.py --out data/hls4ml5_n16 --nobj 16
"""
import argparse, glob, os, time, warnings
import h5py
import numpy as np

CLASS_ORDER = ("g", "q", "w", "z", "t")
JET_LABEL_NAMES = tuple(f"j_{c}" for c in CLASS_ORDER)
PART_NAMES = {"E": "j1_e", "px": "j1_px", "py": "j1_py", "pz": "j1_pz", "pt": "j1_pt"}
FALLBACK_PART_IDX = {"E": 3, "px": 0, "py": 1, "pz": 2, "pt": 5}
CHUNK = 10000


def _names(f, key):
    if key not in f:
        return None
    return [x.decode() if isinstance(x, bytes) else str(x) for x in f[key][:]]


def resolve_columns(f):
    """Return (pmu_cols [E,px,py,pz], pt_col, label_cols [g,q,w,z,t]) for a raw file,
    asserting against the stored feature names when present."""
    pnames = _names(f, "particleFeatureNames")
    jnames = _names(f, "jetFeatureNames")
    njf = f["jets"].shape[1]
    if pnames is None:
        warnings.warn("particleFeatureNames absent; using fixed column indices")
        pidx = dict(FALLBACK_PART_IDX)
    else:
        pidx = {k: pnames.index(v) for k, v in PART_NAMES.items()}
        assert pidx == FALLBACK_PART_IDX, f"unexpected particle columns {pidx}"
    if jnames is None:
        warnings.warn("jetFeatureNames absent; using last 6 jet columns as one-hot")
        lcols = list(range(njf - 6, njf - 1))
    else:
        lcols = [jnames.index(n) for n in JET_LABEL_NAMES]
        assert lcols == list(range(njf - 6, njf - 1)), f"unexpected label cols {lcols}"
    pmu_cols = [pidx["E"], pidx["px"], pidx["py"], pidx["pz"]]
    return pmu_cols, pidx["pt"], lcols


def process_block(const, onehot, nobj, pt_min=None, pmu_cols=(3, 0, 1, 2), pt_col=5):
    """const: (n, P, F) raw constituents; onehot: (n, 5).
    Returns Pmu (n, nobj, 4) float64, Pjet (n, 4) float64 (sum of all post-cut
    constituents, before truncation), Nobj int16, label int8, is_signal int8."""
    pmu = const[:, :, list(pmu_cols)]
    pt = const[:, :, pt_col].copy()
    if pt_min is not None:
        cut = pt <= pt_min
        pmu = np.where(cut[..., None], 0.0, pmu)
        pt = np.where(cut, 0.0, pt)
    pjet = pmu.sum(axis=1)  # full jet: all post-cut constituents, before truncation
    order = np.argsort(-pt, axis=1, kind="stable")
    pmu = np.take_along_axis(pmu, order[..., None], axis=1)
    n_obj = (pmu[:, :, 0] != 0).sum(axis=1).astype(np.int16)
    P = pmu.shape[1]
    if nobj <= P:
        pmu = pmu[:, :nobj, :]
    else:
        pmu = np.concatenate([pmu, np.zeros((pmu.shape[0], nobj - P, 4), pmu.dtype)], axis=1)
    sums = onehot.sum(axis=1)
    if not np.all(sums == 1.0):
        bad = np.nonzero(sums != 1.0)[0]
        raise ValueError(f"one-hot label rows do not sum to 1 (rows {bad[:10]}...)")
    label = np.argmax(onehot, axis=1).astype(np.int8)
    is_signal = (label == 4).astype(np.int8)
    return pmu, pjet, n_obj, label, is_signal


def convert_file(path, nobj, pt_min=None):
    """Read one raw file and return process_block outputs."""
    with h5py.File(path, "r") as f:
        pmu_cols, pt_col, lcols = resolve_columns(f)
        const = f["jetConstituentList"][...]
        onehot = f["jets"][...][:, lcols]
    return process_block(const, onehot, nobj, pt_min, pmu_cols, pt_col)


class _Writer:
    """Pre-sized chunked output file for one split."""

    def __init__(self, path, n, nobj, dtype):
        self.path, self.n, self.nobj, self.pos, self.over = path, n, nobj, 0, 0
        self.counts = np.zeros(len(CLASS_ORDER), np.int64)
        self.g = h5py.File(path, "w")
        ch = max(1, min(CHUNK, n))
        self.d = {
            "Pmu": self.g.create_dataset("Pmu", (n, nobj, 4), dtype=dtype, chunks=(ch, nobj, 4)),
            "Pjet": self.g.create_dataset("Pjet", (n, 4), dtype=dtype, chunks=(ch, 4)),
            "Nobj": self.g.create_dataset("Nobj", (n,), dtype=np.int16, chunks=(ch,)),
            "label": self.g.create_dataset("label", (n,), dtype=np.int8, chunks=(ch,)),
            "is_signal": self.g.create_dataset("is_signal", (n,), dtype=np.int8, chunks=(ch,)),
        }

    def append(self, pmu, pjet, n_obj, label, is_signal):
        s, e = self.pos, self.pos + len(label)
        self.d["Pmu"][s:e] = pmu.astype(self.d["Pmu"].dtype)
        self.d["Pjet"][s:e] = pjet.astype(self.d["Pjet"].dtype)
        self.d["Nobj"][s:e] = n_obj
        self.d["label"][s:e] = label
        self.d["is_signal"][s:e] = is_signal
        self.pos = e
        self.over += int((n_obj > self.nobj).sum())
        self.counts += np.bincount(label, minlength=len(CLASS_ORDER))

    def close(self, attrs):
        assert self.pos == self.n, f"{self.path}: wrote {self.pos} of {self.n} jets"
        for k, v in attrs.items():
            self.g.attrs[k] = v
        self.g.attrs["jets_with_more_than_nobj"] = self.over
        self.g.attrs["n_jets"] = self.n
        self.g.close()


def _list_files(src, max_files):
    files = sorted(glob.glob(os.path.join(src, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {src!r}")
    return files[:max_files] if max_files else files


def _n_jets(path):
    with h5py.File(path, "r") as f:
        return f["jets"].shape[0]


def build(src, files, out_paths, nobj, pt_min, valid_every, dtype):
    """out_paths: dict of split -> final path ('test') or ('valid','train').
    Splits absent from out_paths are not written."""
    t0 = time.time()
    sizes = [_n_jets(p) for p in files]
    total = sum(sizes)
    if "test" in out_paths:
        n_for = {"test": total}
        route = None
    else:
        n_valid = len(range(0, total, valid_every))
        n_for = {"valid": n_valid, "train": total - n_valid}
        route = valid_every
    split_rule = ("all --src-val jets, source-file order" if route is None else
                  f"--src-train jets in sorted-file order; global index % {valid_every} == 0 -> "
                  f"valid, rest -> train")
    attrs = dict(source_dir=os.path.abspath(src),
                 source_files="\n".join(os.path.basename(p) for p in files),
                 nobj=nobj, pt_min=-1.0 if pt_min is None else float(pt_min),
                 valid_every=valid_every, split_rule=split_rule,
                 class_order=",".join(CLASS_ORDER),
                 pmu_columns="E,px,py,pz (from raw cols 3,0,1,2)",
                 pjet_rule="sum of all (post-cut) constituents, before truncation",
                 is_signal_rule="label==4 (top)", pmu_dtype=str(np.dtype(dtype)))
    writers = {s: _Writer(p + ".tmp", n_for[s], nobj, dtype) for s, p in out_paths.items()}
    gidx = 0
    for i, path in enumerate(files):
        out = convert_file(path, nobj, pt_min)
        n = len(out[3])
        if route is None:
            writers["test"].append(*out)
        else:
            is_v = (np.arange(gidx, gidx + n) % route) == 0
            for s, m in (("valid", is_v), ("train", ~is_v)):
                if s in writers:
                    writers[s].append(*(a[m] for a in out))
        gidx += n
        print(f"  [{i+1}/{len(files)}] {os.path.basename(path)}: {n} jets "
              f"({time.time()-t0:.0f}s)", flush=True)
    for s, w in writers.items():
        w.close(attrs)
        os.rename(w.path, out_paths[s])
        cc = ", ".join(f"{c}={k}" for c, k in zip(CLASS_ORDER, w.counts))
        print(f"wrote {out_paths[s]}: {w.n} jets [{cc}], {w.over} "
              f"({100*w.over/max(w.n,1):.1f}%) had >{nobj} constituents, "
              f"{time.time()-t0:.1f}s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--src-train", default="../train 2")
    p.add_argument("--src-val", default="../val 2", help="raw val files -> TEST split")
    p.add_argument("--out", default="data/hls4ml5_n16")
    p.add_argument("--nobj", type=int, default=16)
    p.add_argument("--pt-min", type=float, default=None)
    p.add_argument("--valid-every", type=int, default=10)
    p.add_argument("--max-files", type=int, default=None, help="DEBUG: first k files per split")
    p.add_argument("--dtype", default="float32", help="Pmu storage dtype (float32|float64)")
    p.add_argument("--splits", default="test,valid,train")
    a = p.parse_args(argv)
    dtype = np.dtype(a.dtype)
    os.makedirs(a.out, exist_ok=True)
    want = [s.strip() for s in a.splits.split(",") if s.strip()]
    bad = set(want) - {"test", "valid", "train"}
    if bad:
        p.error(f"unknown splits {sorted(bad)}")
    todo = {}
    for s in want:
        dst = os.path.join(a.out, f"{s}.h5")
        if os.path.exists(dst):
            print(f"skip {dst} (exists)")
        else:
            todo[s] = dst
    if "test" in todo:
        build(a.src_val, _list_files(a.src_val, a.max_files), {"test": todo["test"]},
              a.nobj, a.pt_min, a.valid_every, dtype)
    tv = {s: todo[s] for s in ("valid", "train") if s in todo}
    if tv:
        build(a.src_train, _list_files(a.src_train, a.max_files), tv,
              a.nobj, a.pt_min, a.valid_every, dtype)


if __name__ == "__main__":
    main()
