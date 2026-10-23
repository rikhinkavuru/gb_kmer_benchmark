"""Core, dependency-light building blocks for the PSM-sensitivity analysis (no torch).

* matching / selection families (all operate WITHIN one split, on propensity p or logit):
    hist_temper   -- tempered propensity-histogram matching; alpha=1 reproduces the paper's
                     20-bin proportional match (run_composition_clean.match_proportional)
                     bin-for-bin; alpha=0 keeps every negative.
    trim_band     -- common-support trimming: keep negatives whose logit-propensity lies in the
                     positives' central [q, 1-q] quantile band.
    caliper_1to1  -- two-sided greedy 1:1 nearest-neighbour matching on the logit propensity,
                     without replacement, caliper = c * SD(logit) (Austin 2011).
    random_take   -- nested uniform-random subsets (size-matched controls).
* weights: overlap (pos 1-e, neg e; Li, Morgan & Zaslavsky 2018) and ATT odds (pos 1,
  neg e/(1-e)).
* metrics: vectorised 1000-resample percentile bootstrap (test sequence = resampling unit,
  rng.randint(0, n, n) per resample, exactly the paper's protocol) of AUROC, MCC, balanced
  accuracy, class-balance-weighted MCC (default and prior-corrected threshold), PR-AUC of
  both classes, plus MCC on 20 explicit majority-subsampled class-balanced draws.
"""
import numpy as np

EPS = 1e-12


# =============================================================== selection
def _bins(p_a, p_b, n_bins):
    """Equal-width bins over the pooled range (identical to match_proportional)."""
    lo = float(min(p_a.min(), p_b.min()))
    hi = float(max(p_a.max(), p_b.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    ba = np.clip(np.digitize(p_a, edges) - 1, 0, n_bins - 1)
    bb = np.clip(np.digitize(p_b, edges) - 1, 0, n_bins - 1)
    return ba, bb


def hist_temper(p_target, p_pool, n_bins, alpha, priority):
    """Keep pool members (e.g. negatives) so their propensity histogram moves from the pool's own
    shape (alpha=0) to the target's shape (alpha=1), geometrically in between.

    Per bin b: T_b target count, N_b pool count, f = min_b N_b/T_b (over bins with both > 0).
    kept_b(alpha) = min(N_b, round(N_b * (f*T_b/N_b)**alpha)); bins with T_b=0 are outside
    common support and are dropped for every alpha>0 (as in the paper). At alpha=1 this is
    min(N_b, round(f*T_b)), i.e. the paper's match_proportional count, bin for bin.
    Within a bin, members are taken in ascending `priority` (a fixed random key per member),
    so kept sets are NESTED across alpha. Returns sorted kept indices into p_pool."""
    bt, bp = _bins(p_target, p_pool, n_bins)
    T = np.bincount(bt, minlength=n_bins)
    N = np.bincount(bp, minlength=n_bins)
    ok = (T > 0) & (N > 0)
    f = float(np.min(N[ok] / T[ok])) if ok.any() else 0.0
    keep = []
    for b in range(n_bins):
        if N[b] == 0:
            continue
        members = np.where(bp == b)[0]
        if alpha == 0:
            take = N[b]
        elif T[b] == 0:
            take = 0
        else:
            take = min(N[b], int(round(N[b] * (f * T[b] / N[b]) ** alpha)))
        if take > 0:
            keep.extend(members[np.argsort(priority[members], kind="stable")[:take]])
    return np.array(sorted(keep), dtype=int)


def trim_band(l_pos, l_neg, q):
    """Keep negatives whose logit-propensity is inside the positives' [q, 1-q] quantile band."""
    lo, hi = np.quantile(l_pos, q), np.quantile(l_pos, 1.0 - q)
    return np.where((l_neg >= lo) & (l_neg <= hi))[0]


def caliper_1to1(l_pos, l_neg, caliper, rng):
    """Greedy 1:1 nearest-neighbour matching without replacement on a 1-D score.
    Positives are processed in random order (rng); each takes the closest still-available
    negative if |diff| <= caliper. Returns (pos_idx, neg_idx) of matched pairs."""
    order = np.argsort(l_neg, kind="stable")
    s = l_neg[order]
    n = len(s)
    right = np.arange(n + 1)            # next available >= i  (n = sentinel "none")
    left = np.arange(-1, n)             # left[i+1] = previous available <= i (-1 sentinel)

    def find_r(i):
        root = i
        while right[root] != root:
            root = right[root]
        while right[i] != root:
            right[i], i = root, right[i]
        return root

    def find_l(i):                      # i in [-1, n-1]; stored at left[i+1]
        root = i
        while left[root + 1] != root:
            root = left[root + 1]
        while left[i + 1] != root:
            left[i + 1], i = root, left[i + 1]
        return root

    pos_m, neg_m = [], []
    for pi in rng.permutation(len(l_pos)):
        x = l_pos[pi]
        j = int(np.searchsorted(s, x))
        r = find_r(j) if j <= n else n
        lft = find_l(j - 1)
        best, bd = -1, np.inf
        if r < n and abs(s[r] - x) < bd:
            best, bd = r, abs(s[r] - x)
        if lft >= 0 and abs(s[lft] - x) < bd:
            best, bd = lft, abs(s[lft] - x)
        if best >= 0 and bd <= caliper:
            pos_m.append(pi)
            neg_m.append(order[best])
            right[best] = best + 1       # remove `best`
            left[best + 1] = best - 1
    return np.array(pos_m, dtype=int), np.array(neg_m, dtype=int)


def random_take(n_pool, k, priority):
    """Nested uniform-random subset: the k members with the smallest random priority."""
    return np.sort(np.argsort(priority, kind="stable")[:k]) if k > 0 else np.array([], dtype=int)


# ================================================================= weights
def overlap_weights(e, y):
    return np.where(y == 1, 1.0 - e, e)


def att_weights(e, y):
    return np.where(y == 1, 1.0, e / np.clip(1.0 - e, EPS, None))


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / max((w ** 2).sum(), EPS))


def smd(a, b, wa=None, wb=None):
    """Per-feature standardized mean difference (a - b), optionally weighted. The denominator is the
    UNWEIGHTED average-variance SD of the two sets passed in, i.e. of each arm's KEPT sequences (not the
    pre-matching SD of Austin's matched-sample convention), so an arm's SMD is scaled by its own
    spread. max|SMD| is used descriptively only (no decision depends on it)."""
    wa = np.ones(len(a)) if wa is None else np.asarray(wa, float)
    wb = np.ones(len(b)) if wb is None else np.asarray(wb, float)
    ma = (a * wa[:, None]).sum(0) / wa.sum()
    mb = (b * wb[:, None]).sum(0) / wb.sum()
    sd = np.sqrt((a.var(0) + b.var(0)) / 2.0)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (ma - mb) / sd


# ================================================================= metrics
def _grouped(scores, ascending=True):
    order = np.argsort(scores if ascending else -scores, kind="mergesort")
    ss = scores[order]
    starts = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1]
    return order, starts


def auc_rows(W, y, scores):
    """AUROC for every row of weight matrix W (B x n), ties counted 1/2 (== roc_auc_score)."""
    order, starts = _grouped(scores, True)
    Ws = W[:, order]
    ys = y[order].astype(bool)
    pos = np.add.reduceat(np.where(ys, Ws, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(ys, 0.0, Ws), starts, axis=1)
    cneg = np.cumsum(neg, axis=1) - neg
    P, N = pos.sum(1), neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (pos * (cneg + 0.5 * neg)).sum(1) / (P * N)
    a[(P <= 0) | (N <= 0)] = np.nan
    return a


def ap_rows(W, y, scores):
    """Average precision (== sklearn average_precision_score, step-wise) per row of W."""
    order, starts = _grouped(scores, False)
    Ws = W[:, order]
    ys = y[order].astype(bool)
    pos = np.add.reduceat(np.where(ys, Ws, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(ys, 0.0, Ws), starts, axis=1)
    tp, fp = np.cumsum(pos, 1), np.cumsum(neg, 1)
    P = pos.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        a = (prec * pos).sum(1) / P
    a[P <= 0] = np.nan
    return a


def conf_rows(W, y, pred, balance=False):
    """Weighted confusion counts per row; with balance=True each class is rescaled to total 1."""
    yb, pb = y.astype(bool), pred.astype(bool)
    tp = W @ (yb & pb).astype(float)
    fn = W @ (yb & ~pb).astype(float)
    fp = W @ (~yb & pb).astype(float)
    tn = W @ (~yb & ~pb).astype(float)
    if balance:
        P, N = tp + fn, fp + tn
        with np.errstate(invalid="ignore", divide="ignore"):
            tp, fn, fp, tn = tp / P, fn / P, fp / N, tn / N
    return tp, fn, fp, tn


def mcc_from(tp, fn, fp, tn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)   # sklearn: 0 if undefined
    m[~np.isfinite(tp + fn + fp + tn)] = np.nan
    return m


def bacc_from(tp, fn, fp, tn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def _all_metrics(W, y, p1, thr_pc):
    pred = (p1 > 0.5).astype(int)            # == argmax of [1-p, p] (ties -> class 0)
    pred_pc = (p1 > thr_pc).astype(int)
    c = conf_rows(W, y, pred)
    cb = conf_rows(W, y, pred, balance=True)
    cbp = conf_rows(W, y, pred_pc, balance=True)
    return dict(auroc=auc_rows(W, y, p1), mcc=mcc_from(*c), bacc=bacc_from(*c),
                mcc_bw=mcc_from(*cb), mcc_bw_pc=mcc_from(*cbp), bacc_pc=bacc_from(*cbp),
                ap_pos=ap_rows(W, y, p1), ap_neg=ap_rows(W, 1 - y, -p1))


METRICS = ["auroc", "mcc", "bacc", "mcc_bw", "mcc_bw_pc", "bacc_pc", "ap_pos", "ap_neg"]


def evaluate(y, p1, w=None, thr_pc=0.5, B=1000, seed=42, n_bal=20, chunk=250):
    """Point metrics + 95% percentile bootstrap CIs (B resamples of test sequences, weights
    carried) + MCC on n_bal explicit class-balanced draws (unweighted evaluations only).

    thr_pc = prior-corrected threshold = positive fraction of the TRAINING set (the
    Bayes threshold for balanced accuracy under calibrated scores); training-only info."""
    y = np.asarray(y, np.int64)
    p1 = np.asarray(p1, np.float64)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, np.float64)
    out = {}
    pt = _all_metrics(w[None, :], y, p1, thr_pc)
    for k in METRICS:
        out[k] = float(pt[k][0])
    wp, wn = w[y == 1], w[y == 0]
    out.update(n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
               prev_pos=float(wp.sum() / w.sum()), ess_pos=ess(wp), ess_neg=ess(wn))
    if B and B > 0:
        rng = np.random.RandomState(seed)
        acc = {k: [] for k in METRICS}
        done = 0
        while done < B:
            bsz = min(chunk, B - done)
            Wc = np.empty((bsz, n))
            for i in range(bsz):
                Wc[i] = np.bincount(rng.randint(0, n, n), minlength=n)
            Wc *= w[None, :]
            m = _all_metrics(Wc, y, p1, thr_pc)
            for k in METRICS:
                acc[k].append(m[k])
            done += bsz
        for k in METRICS:
            v = np.concatenate(acc[k])
            out[k + "_lo"] = float(np.nanpercentile(v, 2.5))
            out[k + "_hi"] = float(np.nanpercentile(v, 97.5))
    # explicit class-balanced draws (majority class subsampled to minority size)
    if n_bal and np.allclose(w, 1.0):
        rng = np.random.RandomState(seed + 1)
        ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
        m = min(len(ip), len(ineg))
        vals, vals_pc = [], []
        for _ in range(n_bal):
            sel = np.r_[rng.choice(ip, m, replace=False), rng.choice(ineg, m, replace=False)]
            ones = np.ones((1, len(sel)))
            vals.append(mcc_from(*conf_rows(ones, y[sel], (p1[sel] > 0.5).astype(int)))[0])
            vals_pc.append(mcc_from(*conf_rows(ones, y[sel], (p1[sel] > thr_pc).astype(int)))[0])
        out.update(mcc_bal20=float(np.mean(vals)), mcc_bal20_sd=float(np.std(vals, ddof=1)),
                   mcc_bal20_pc=float(np.mean(vals_pc)), mcc_bal20_pc_sd=float(np.std(vals_pc, ddof=1)),
                   n_bal_per_class=int(m))
    else:
        out.update(mcc_bal20=np.nan, mcc_bal20_sd=np.nan, mcc_bal20_pc=np.nan,
                   mcc_bal20_pc_sd=np.nan, n_bal_per_class=np.nan)
    return out


# ============================================================ out-of-model features
def dust_lowcomplexity_frac(seqs, win=64, step=4, thr=2.0, cells=1_000_000):
    """Fraction of bases covered by at least one low-complexity window (DUST-like triplet score
    S = sum_t c_t(c_t-1)/2 / (l-1), l = #triplets in window; S > 2.0 ~ sdust T=20).
    Chunked so that (#seqs x max length) <= `cells` (keeps RAM ~150 MB)."""
    import motif_match
    seqs = list(seqs)
    out = np.zeros(len(seqs))
    maxlen = max(len(x) for x in seqs)
    chunk = max(20, cells // max(maxlen, 1))
    for s0 in range(0, len(seqs), chunk):
        sub = seqs[s0:s0 + chunk]
        codes = motif_match.encode_sequences(list(sub)).astype(np.int16)   # (m, S), 4 = N/pad
        m, S = codes.shape
        lens = np.array([len(x) for x in sub])
        valid3 = (codes[:, :-2] < 4) & (codes[:, 1:-1] < 4) & (codes[:, 2:] < 4)
        trip = np.where(valid3, codes[:, :-2] * 16 + codes[:, 1:-1] * 4 + codes[:, 2:], 64)
        l = win - 2
        starts = np.arange(0, max(S - win + 1, 1), step)
        masked = np.zeros((m, S), bool)
        onehot_cum = np.zeros((m, trip.shape[1] + 1, 65), np.int16)
        np.cumsum(np.eye(65, dtype=np.int16)[trip], axis=1, out=onehot_cum[:, 1:, :])
        for st in starts:
            c = onehot_cum[:, st + l, :64] - onehot_cum[:, st, :64]
            sc = (c * (c - 1) / 2).sum(1) / (l - 1)
            hit = (sc > thr) & (st + win <= lens)
            masked[hit, st:st + win] = True
        out[s0:s0 + chunk] = masked.sum(1) / np.maximum(lens, 1)
    return out
