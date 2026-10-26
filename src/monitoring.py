"""Model-monitoring helpers: PSI (score drift) and CSI (feature drift).

PSI = sum((actual% - expected%) * ln(actual% / expected%)) over bins fixed on the *expected* (reference) sample.
Rule of thumb: < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant shift.
"""
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10, eps=1e-4):
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to value-share comparison
        vals = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def csi_table(ref: pd.DataFrame, cur: pd.DataFrame, cols, bins=10):
    """Characteristic Stability Index per feature (PSI applied to each input)."""
    rows = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(ref[c]):
            r = ref[c].value_counts(normalize=True)
            a = cur[c].value_counts(normalize=True).reindex(r.index).fillna(0)
            r, a = r.clip(lower=1e-4), a.clip(lower=1e-4)
            v = float(((a - r) * np.log(a / r)).sum())
        else:
            v = psi(ref[c], cur[c], bins)
        rows.append((c, v))
    return pd.DataFrame(rows, columns=["feature", "csi"]).sort_values("csi", ascending=False)


def psi_status(v):
    return "stable" if v < 0.10 else ("investigate" if v < 0.25 else "significant shift")
