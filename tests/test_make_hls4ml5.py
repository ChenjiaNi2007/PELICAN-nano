"""Tests for scripts/make_hls4ml5.py on a tiny synthetic raw hls4ml dataset."""
import os
import h5py
import numpy as np
import pytest

from scripts.make_hls4ml5 import main, process_block, convert_file

PART_NAMES = ["j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel", "j1_eta",
              "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot", "j1_deltaR",
              "j1_costheta", "j1_costhetarel", "j1_pdgid"]
JET_NAMES = [f"j_dummy{i}" for i in range(53)] + ["j_g", "j_q", "j_w", "j_z", "j_t", "j_undef"]
NJ, P, NOBJ = 25, 150, 16


def _write_raw(path, seed):
    rng = np.random.default_rng(seed)
    mult = rng.integers(3, 40, size=NJ)
    mult[0], mult[1] = 150, 5          # include full and short jets
    c = np.zeros((NJ, P, 16))
    for j, m in enumerate(mult):
        pt = np.sort(rng.uniform(0.5, 100.0, m))[::-1]
        c[j, :m, 5] = pt
        c[j, :m, 0] = rng.normal(size=m)          # px
        c[j, :m, 1] = rng.normal(size=m)          # py
        c[j, :m, 2] = rng.normal(size=m)          # pz
        c[j, :m, 3] = pt + rng.uniform(1, 2, m)   # E (nonzero)
        c[j, :m, 6:] = rng.normal(size=(m, 10))
    jets = rng.normal(size=(NJ, 59))
    lab = rng.integers(0, 5, size=NJ)
    jets[:, -6:] = 0.0
    jets[np.arange(NJ), 53 + lab] = 1.0
    with h5py.File(path, "w") as f:
        f["jetConstituentList"] = c
        f["jets"] = jets
        f["jetFeatureNames"] = np.array([n.encode() for n in JET_NAMES], dtype=object)
        f["particleFeatureNames"] = np.array([n.encode() for n in PART_NAMES], dtype=object)
        f["jetImage"] = np.zeros((NJ, 2, 2))
    return c, lab, mult


@pytest.fixture
def raw(tmp_path):
    tr, va = tmp_path / "train 2", tmp_path / "val 2"
    tr.mkdir(); va.mkdir()
    # names chosen so sorted order != creation order
    t_b = _write_raw(tr / "jetImage_b.h5", 1)
    t_a = _write_raw(tr / "jetImage_a.h5", 2)
    v = _write_raw(va / "jetImage_v.h5", 3)
    return tmp_path, [t_a, t_b], v   # train data in SORTED order


def _expected_pmu(c, nobj=NOBJ):
    return c[:, :nobj, [3, 0, 1, 2]].astype(np.float32)


def _run(tmp, out, *extra):
    main(["--src-train", str(tmp / "train 2"), "--src-val", str(tmp / "val 2"),
          "--out", str(out), "--nobj", str(NOBJ), "--valid-every", "4", *extra])


def _load(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][...] for k in f.keys()}, dict(f.attrs)


def test_full_conversion(raw):
    tmp, train, (vc, vlab, vmult) = raw
    out = tmp / "out"
    _run(tmp, out)
    assert sorted(os.listdir(out)) == ["test.h5", "train.h5", "valid.h5"]

    # test split: source order, column mapping, labels, Nobj
    d, attrs = _load(out / "test.h5")
    assert set(d) == {"Pmu", "Pjet", "Nobj", "label", "is_signal"}
    assert d["Pmu"].shape == (NJ, NOBJ, 4) and d["Pmu"].dtype == np.float32
    assert d["Nobj"].dtype == np.int16 and d["label"].dtype == np.int8
    np.testing.assert_array_equal(d["Pmu"], _expected_pmu(vc))
    np.testing.assert_array_equal(d["label"], vlab)
    np.testing.assert_array_equal(d["is_signal"], (vlab == 4).astype(np.int8))
    np.testing.assert_array_equal(d["Nobj"], vmult)
    # zero padding for short jet (mult 5)
    assert np.all(d["Pmu"][1, 5:] == 0) and np.all(d["Pmu"][1, :5, 0] != 0)
    for k in ["source_dir", "source_files", "nobj", "pt_min", "valid_every", "split_rule",
              "class_order", "pmu_columns", "is_signal_rule", "pmu_dtype",
              "jets_with_more_than_nobj", "n_jets"]:
        assert k in attrs, k
    assert attrs["class_order"] == "g,q,w,z,t" and attrs["pt_min"] == -1.0
    assert attrs["n_jets"] == NJ
    assert attrs["jets_with_more_than_nobj"] == int((vmult > NOBJ).sum())

    # every-4th global index -> valid, rest -> train
    allc = np.concatenate([t[0] for t in train])
    alllab = np.concatenate([t[1] for t in train])
    allmult = np.concatenate([t[2] for t in train])
    g = np.arange(len(allc))
    isv = g % 4 == 0
    dv, av = _load(out / "valid.h5")
    dt, at = _load(out / "train.h5")
    assert len(dv["label"]) == isv.sum() == 13 and len(dt["label"]) == (~isv).sum() == 37
    np.testing.assert_array_equal(dv["Pmu"], _expected_pmu(allc[isv]))
    np.testing.assert_array_equal(dt["Pmu"], _expected_pmu(allc[~isv]))
    np.testing.assert_array_equal(dv["label"], alllab[isv])
    np.testing.assert_array_equal(dt["Nobj"], allmult[~isv])
    assert at["source_files"] == "jetImage_a.h5\njetImage_b.h5"
    assert av["valid_every"] == 4

    # skip-existing
    mtime = os.path.getmtime(out / "test.h5")
    _run(tmp, out)
    assert os.path.getmtime(out / "test.h5") == mtime


def test_pt_sorted_and_truncated(raw):
    tmp, _, (vc, _, _) = raw
    pmu, _, n_obj, _, _ = convert_file(str(tmp / "val 2" / "jetImage_v.h5"), NOBJ)
    assert pmu.shape == (NJ, NOBJ, 4)
    # E = pT + const offset in synthetic data is not monotone; check pT order via source
    pt = vc[:, :NOBJ, 5]
    assert np.all(np.diff(pt, axis=1) <= 0)
    np.testing.assert_array_equal(pmu, vc[:, :NOBJ, [3, 0, 1, 2]])


def test_pt_min_cut(raw):
    tmp, _, (vc, vlab, _) = raw
    cut = 30.0
    out = tmp / "cut"
    _run(tmp, out, "--pt-min", str(cut), "--splits", "test")
    d, attrs = _load(out / "test.h5")
    assert attrs["pt_min"] == cut
    for j in range(NJ):
        keep = (vc[j, :, 3] != 0) & (vc[j, :, 5] > cut)
        exp = vc[j, keep][:, [3, 0, 1, 2]]
        assert d["Nobj"][j] == keep.sum()
        k = min(len(exp), NOBJ)
        np.testing.assert_array_equal(d["Pmu"][j, :k], exp[:k].astype(np.float32))
        assert np.all(d["Pmu"][j, k:] == 0)


def test_pt_min_packs_front_when_unsorted():
    # a low-pT particle in the middle: survivors must pack to the front, stable
    c = np.zeros((1, 6, 16))
    c[0, :4, 5] = [50.0, 1.0, 40.0, 30.0]
    c[0, :4, 3] = [5.0, 6.0, 7.0, 8.0]
    oh = np.array([[0, 0, 0, 0, 1.0]])
    pmu, _, n_obj, label, sig = process_block(c, oh, 5, pt_min=2.0)
    np.testing.assert_array_equal(pmu[0, :, 0], [5.0, 7.0, 8.0, 0.0, 0.0])
    assert n_obj[0] == 3 and label[0] == 4 and sig[0] == 1


def test_nobj_larger_than_slots_pads():
    c = np.zeros((2, 3, 16)); c[:, :2, 3] = 1.0; c[:, :2, 5] = [2.0, 1.0]
    oh = np.eye(5)[[0, 2]]
    pmu, _, n_obj, label, _ = process_block(c, oh, 5)
    assert pmu.shape == (2, 5, 4) and np.all(pmu[:, 2:] == 0)
    np.testing.assert_array_equal(label, [0, 2]); np.testing.assert_array_equal(n_obj, [2, 2])


def test_bad_onehot_raises():
    c = np.zeros((1, 3, 16))
    with pytest.raises(ValueError):
        process_block(c, np.array([[0, 0, 0, 0, 0.0]]), 3)


def test_pjet_full_jet_sum(raw):
    """Pjet = sum of ALL (post-cut) constituents' (E,px,py,pz), before truncation."""
    tmp, _, (vc, _, vmult) = raw
    assert vmult.max() > NOBJ  # truncation actually drops constituents
    out = tmp / "out"
    _run(tmp, out, "--splits", "test")
    d, attrs = _load(out / "test.h5")
    assert d["Pjet"].shape == (NJ, 4) and d["Pjet"].dtype == np.float32
    assert attrs["pjet_rule"] == "sum of all (post-cut) constituents, before truncation"
    exp = vc[:, :, [3, 0, 1, 2]].sum(axis=1).astype(np.float32)
    np.testing.assert_allclose(d["Pjet"], exp, rtol=1e-6, atol=1e-5)
    # differs from the truncated sum whenever the jet has more than NOBJ constituents
    trunc = d["Pmu"].astype(np.float64).sum(axis=1)
    assert np.all(np.abs(d["Pjet"][vmult > NOBJ, 0] - trunc[vmult > NOBJ, 0]) > 1e-3)

    cut = 30.0
    out2 = tmp / "cut_pjet"
    _run(tmp, out2, "--pt-min", str(cut), "--splits", "test")
    d2, _ = _load(out2 / "test.h5")
    keep = (vc[:, :, 3] != 0) & (vc[:, :, 5] > cut)
    exp2 = np.where(keep[..., None], vc[:, :, [3, 0, 1, 2]], 0.0).sum(axis=1).astype(np.float32)
    np.testing.assert_allclose(d2["Pjet"], exp2, rtol=1e-6, atol=1e-5)
