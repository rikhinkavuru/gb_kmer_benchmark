"""Altschul-Erikson dinucleotide-preserving shuffle (pure Python, CPU, deterministic).

Given a sequence, returns a permutation of its letters that EXACTLY preserves the dinucleotide
(and therefore mononucleotide / GC) composition, sampled near-uniformly among all such permutations
via the Eulerian-path method (Altschul & Erikson, 1985). Used by Upgrade 5 to build
composition-matched negatives from positives, so that any residual class signal is higher-order
(motif) structure, not composition. No external uShuffle dependency.
"""


def dinuc_shuffle(seq, rng, max_tries=25):
    """Return a dinucleotide-preserving shuffle of ``seq`` using ``rng`` (np.random.RandomState).
    Falls back to the original sequence if no valid Eulerian arborescence is found in max_tries
    (extremely rare for real DNA); callers can detect this via verify_dinuc_preserved."""
    s = str(seq)
    n = len(s)
    if n < 4 or len(set(s)) < 2:
        return s
    end = s[-1]
    out = {}
    for i in range(n - 1):
        out.setdefault(s[i], []).append(s[i + 1])
    verts = set(s)

    for _ in range(max_tries):
        # 1. choose one "last-exit" edge index per vertex != end
        last_exit = {}
        ok = True
        for v in verts:
            if v == end:
                continue
            if not out.get(v):
                ok = False
                break
            last_exit[v] = rng.randint(len(out[v]))
        if not ok:
            continue
        # 2. the last-exit edges must form an arborescence into `end` (every vertex reaches end)
        good = True
        for v in verts:
            if v == end:
                continue
            seen, cur = set(), v
            while cur != end:
                if cur in seen or cur not in last_exit:
                    good = False
                    break
                seen.add(cur)
                cur = out[cur][last_exit[cur]]
            if not good:
                break
        if not good:
            continue
        # 3. order each vertex's edges: shuffle the non-last-exit edges, last-exit edge goes LAST
        order = {}
        for v in verts:
            m = len(out[v])
            if v == end:
                idxs = list(range(m)); _shuffle(idxs, rng); order[v] = idxs
            else:
                le = last_exit[v]
                rest = [i for i in range(m) if i != le]
                _shuffle(rest, rng); order[v] = rest + [le]
        # 4. Eulerian traversal from s[0]
        res = [s[0]]
        ptr = {v: 0 for v in verts}
        cur = s[0]
        for _ in range(n - 1):
            ei = order[cur][ptr[cur]]; ptr[cur] += 1
            cur = out[cur][ei]
            res.append(cur)
        return "".join(res)
    return s


def _shuffle(lst, rng):
    """In-place Fisher-Yates using an np.random.RandomState (deterministic)."""
    for i in range(len(lst) - 1, 0, -1):
        j = rng.randint(i + 1)
        lst[i], lst[j] = lst[j], lst[i]


def dinuc_counts(seq):
    """dict of dinucleotide -> count for the sequence."""
    d = {}
    for i in range(len(seq) - 1):
        k = seq[i:i + 2]
        d[k] = d.get(k, 0) + 1
    return d


def verify_dinuc_preserved(original, shuffled):
    """True iff ``shuffled`` has identical dinucleotide counts to ``original`` (and same length)."""
    return len(original) == len(shuffled) and dinuc_counts(original) == dinuc_counts(shuffled)
