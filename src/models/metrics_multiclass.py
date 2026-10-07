"""
Multi-class (K-logit) metrics for nanoPELICAN with --n-out K > 1.

Mirrors the API of metrics_classifier.py (metrics / minibatch_metrics /
minibatch_metrics_string) so Trainer can use it unchanged.
`compute_multiclass_metrics` is pure numpy/sklearn and is also imported by
standalone evaluation scripts -- keep its signature and key names stable.
"""
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CLASSES = ('g', 'q', 'w', 'z', 't')          # hls4ml jet dataset class order (label 0..4)


def class_names(n_classes: int) -> list:
    if n_classes == len(CLASSES):
        return list(CLASSES)
    return [f'c{i}' for i in range(n_classes)]


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ovr_valid(y):
    """True if a binary one-vs-rest label vector has both positives and negatives."""
    n_pos = int(y.sum())
    return 0 < n_pos < y.shape[0]


def compute_multiclass_metrics(logits, labels, names=None, fpr_points=(0.01, 0.1)) -> dict:
    """Pure numpy/sklearn (no torch). logits: (n, K) float array; labels: (n,) ints in 0..K-1.

    Returns an ordered dict with keys, in this order:
      'accuracy', 'AUC' (macro mean of one-vs-rest AUCs), f'AUC_{c}' per class,
      then f'TPR@FPR{p}_{c}' per class for each p in fpr_points.
    One-vs-rest score = softmax probability of class c. A class with no positives
    or no negatives in `labels` gets AUC/TPR = 0.0 (never raises).
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    K = logits.shape[1]
    if names is None:
        names = class_names(K)
    probs = _softmax(logits)

    out = OrderedDict()
    out['accuracy'] = float((logits.argmax(axis=1) == labels).mean()) if labels.size else 0.0

    aucs, tprs = [], {p: [] for p in fpr_points}
    for k in range(K):
        y = (labels == k).astype(np.int64)
        if _ovr_valid(y):
            aucs.append(float(roc_auc_score(y, probs[:, k])))
            fpr, tpr, _ = roc_curve(y, probs[:, k])
            for p in fpr_points:
                tprs[p].append(float(np.interp(p, fpr, tpr)))
        else:
            aucs.append(0.0)
            for p in fpr_points:
                tprs[p].append(0.0)

    out['AUC'] = float(np.mean(aucs)) if aucs else 0.0
    for c, a in zip(names, aucs):
        out[f'AUC_{c}'] = a
    for p in fpr_points:
        for c, t in zip(names, tprs[p]):
            out[f'TPR@FPR{p}_{c}'] = t
    return out


def metrics(predict, targets, loss_fn, prefix, logger=None):
    loss = loss_fn(predict, targets.long()).item()
    logits = predict.detach().cpu().double().numpy()
    labels = targets.detach().cpu().long().numpy()
    names = class_names(logits.shape[1])
    m = compute_multiclass_metrics(logits, labels, names=names)

    # One-vs-rest ROC curves, one file per class (like the binary module's prefix+'_ROC.csv').
    probs = _softmax(logits)
    for k, c in enumerate(names):
        y = (labels == k).astype(np.int64)
        if _ovr_valid(y):
            fpr, tpr, _ = roc_curve(y, probs[:, k])
            np.savetxt(prefix + f'_ROC_{c}.csv', np.stack([fpr, tpr]), delimiter=',')

    result = {'loss': loss, **m}
    per_class = ' '.join('{} {:.4f}'.format(c, m[f'AUC_{c}']) for c in names)
    string = ' L: {:10.4f}, ACC: {:10.4f}, mAUC: {:10.4f} | AUC {}'.format(loss, m['accuracy'], m['AUC'], per_class)
    return result, string


def minibatch_metrics(predict, targets, loss):
    logits = predict.detach().cpu().double().numpy()
    labels = targets.detach().cpu().long().numpy().reshape(-1)
    accuracy = float((logits.argmax(axis=1) == labels).mean()) if labels.size else 0.0
    probs = _softmax(logits)
    aucs = []
    for k in range(logits.shape[1]):
        y = (labels == k).astype(np.int64)
        if _ovr_valid(y):
            aucs.append(float(roc_auc_score(y, probs[:, k])))
    mauc = float(np.mean(aucs)) if aucs else 0.0
    return [loss, accuracy, mauc]


def minibatch_metrics_string(metrics):
    string = ', L:{:> 9.4f}, ACC:{:> 9.4f}, mAUC:{:> 9.4f}'.format(*metrics)
    return string
