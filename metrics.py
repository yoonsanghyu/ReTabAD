import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve


def get_auroc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def get_auprc(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


def get_f1(y_true, y_pred, threshold=None):
    if threshold is None:
        threshold = get_best_f1_threshold(y_true, y_pred)
    y_pred = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred)


def get_best_f1_threshold(gt, anomaly_scores):
    fpr, tpr, thresholds = roc_curve(gt, anomaly_scores, drop_intermediate=False)
    P, N = (gt == 1).sum(), (gt == 0).sum()
    fp = np.array(fpr * N, dtype=int)
    tn = np.array(N - fp, dtype=int)
    tp = np.array(tpr * P, dtype=int)
    fn = np.array(P - tp, dtype=int)

    den_p = tp+fp
    den_p[den_p==0] = 1
    precision = tp / den_p
    den_r = tp+fn
    den_r[den_r==0] = 1
    recall = tp / den_r

    den = precision + recall
    den[den==0.0] = 1
    f1 = 2 * (precision * recall) / den
    idx = np.argmax(f1)
    best_threshold = thresholds[idx]
    return best_threshold


def get_summary_metrics(y_true, y_pred, threshold=None):
    f1 = get_f1(y_true, y_pred, threshold)
    auroc = get_auroc(y_true, y_pred)
    auprc = get_auprc(y_true, y_pred)
    return {
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc)
    }