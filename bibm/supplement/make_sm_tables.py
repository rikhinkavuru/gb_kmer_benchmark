#!/usr/bin/env python3
"""Generate every table, every in-text number and the figure copies of the B290 Supplementary Material.

Nothing in bibm/supplement/B290_SM.tex is retyped from a report: each table is written here from the
result files named in its caption, and each number quoted in the prose is a LaTeX macro defined here
(generated/sm_numbers.tex; every macro carries a comment naming its source file and column).

Usage (from the repository root, torch-free analysis environment):
    bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python bibm/supplement/make_sm_tables.py
then build:
    cd bibm/supplement && ../lowcpu.sh /opt/homebrew/bin/tectonic B290_SM.tex

Outputs (all under bibm/supplement/):
    generated/tab_*.tex        one file per table (caption names its source files)
    generated/sm_numbers.tex   \\newcommand macros for every number used in the prose
    generated/manifest.tex     table of every input file with its SHA-256 prefix
    figs/*.pdf                 copies of the figures used (originals untouched); two are SM-local re-plots from the
                               result CSVs (route3_control_motif_panel.pdf: the experiment's own panel functions, legend
                               of panel (b) moved off the lines; fm_frozen_probe.pdf: vector, 8-pt text)

Rules: reads result files only; never refits a model; derived quantities are simple arithmetic on
file values and are marked "derived" in the macro comments. Rounding: to the precision shown.
"""
import hashlib
import importlib.metadata as im
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RU = ROOT / "results" / "upgrades"
RR = ROOT / "results"
OUT = HERE / "generated"
FIGS = HERE / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

MANIFEST = {}   # relative path -> (sha256, rows or '')
MACROS = []     # (name, value, source)
MINUS = "\u2212"


# ----------------------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def register(p, rows=""):
    MANIFEST[rel(p)] = (sha256(p), rows)


def load(name, base=RU):
    p = Path(name) if os.path.isabs(str(name)) else base / name
    df = pd.read_csv(p)
    register(p, len(df))
    return df


def read_text(p):
    p = Path(p)
    register(p)
    return p.read_text()


def isnan(x):
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except TypeError:
        return False


def f(x, d=3, sign=False):
    """Number with d decimals, Unicode minus, optional explicit plus."""
    if isnan(x):
        return "--"
    x = float(x)
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0.0:
        return MINUS + s
    if sign and float(s) != 0.0:
        return "+" + s
    return s


def cib(x, d=3, sign=False):
    """A CI bound: like f(), but a nonzero bound that rounds to zero keeps its sign (e.g. -0.0002 -> '-0.000'),
    so a reader can see whether the interval crosses 0."""
    if isnan(x):
        return "--"
    x = float(x)
    s = f"{abs(x):.{d}f}"
    if float(s) == 0.0 and x != 0.0:
        return (MINUS + s) if x < 0 else (("+" + s) if sign else s)
    return f(x, d, sign)


def ci(lo, hi, d=3, sign=False):
    return f"[{cib(lo, d, sign)}, {cib(hi, d, sign)}]"


def pct(x, d=0, sign=False):
    if isnan(x):
        return "--"
    return f(100 * float(x), d, sign) + "\\%"


def prange(lo, hi, d=0):
    return f(100 * float(lo), d) + "--" + f(100 * float(hi), d) + "\\%"


def pci(lo, hi, d=0):
    return f"[{f(100 * lo, d)}, {f(100 * hi, d)}]"


def sig2(p):
    """p-value with two significant digits (0.10, 0.013, 0.0004)."""
    if isnan(p):
        return "--"
    p = float(p)
    if p == 0:
        return "0"
    e = math.floor(math.log10(abs(p)))
    if p < 1e-4:
        m = p / 10 ** e
        return f"{m:.1f}{{\\texttimes}}10\\textsuperscript{{{MINUS}{abs(e)}}}"
    if p < 1e-3:
        return f"{p:.{-e}f}"
    d = max(0, 1 - e)
    return f"{p:.{d}f}"


def intc(n):
    return f"{int(round(float(n))):,}".replace(",", "{,}")


def esc(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("~", "\\textasciitilde{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#").replace("$", "\\$"))


def tt(s):
    return "\\task{" + esc(s) + "}"


def src(*names):
    # a visibly wider gap after the comma: a plain interword space looks lost next to typewriter text
    return ",\\enspace ".join("\\src{" + esc(n) + "}" for n in names)


def macro(name, value, source):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in {m[0] for m in MACROS}, name
    MACROS.append((name, str(value), source))
    return value


SHORT = {
    "human_enhancers_cohn": "cohn", "nt_enhancers": "nt\\_enhancers",
    "drosophila_enhancers_stark": "control", "human_ensembl_regulatory": "regulatory",
    "human_nontata_promoters": "nontata", "human_enhancers_ensembl": "enh\\_ensembl",
    "human_ocr_ensembl": "ocr", "nt_splice_sites_all": "splice",
    "demo_human_or_worm": "human\\_or\\_worm", "demo_coding_vs_intergenomic_seqs": "coding",
    "dummy_mouse_enhancers_ensembl": "mouse\\_enh", "nt_promoter_all": "nt\\_prom\\_all",
    "nt_promoter_no_tata": "nt\\_prom\\_no\\_tata", "nt_promoter_tata": "nt\\_prom\\_tata",
    "nt_enhancers_types": "nt\\_enh\\_types",
}
TASK_ORDER = ["demo_human_or_worm", "human_ensembl_regulatory", "nt_promoter_all", "nt_promoter_no_tata",
              "human_nontata_promoters", "nt_promoter_tata", "demo_coding_vs_intergenomic_seqs",
              "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl", "nt_enhancers",
              "nt_enhancers_types", "human_enhancers_cohn", "human_ocr_ensembl",
              "drosophila_enhancers_stark", "nt_splice_sites_all"]


def short(t):
    v = SHORT.get(t, esc(t))
    return v if v == "control" else "\\task{" + v + "}"


def pth(p):
    """a path that may break after '/' and '_' (LaTeX \\pth, url-style; no escaping needed)"""
    return "\\pth{" + str(p) + "}"


def write_table(fname, label, caption, sources, colspec, header, rows, notes=None,
                size="\\footnotesize", pos="!htbp", tabcolsep=None, tabular="tabular", width=None):
    """rows: list of lists (cells) or strings 'midrule' / ('group', text)."""
    ncol = len(header[-1]) if header and isinstance(header[0], (list, tuple)) else len(header)
    L = [f"% generated by bibm/supplement/make_sm_tables.py -- do not edit; sources: {', '.join(sources)}",
         f"\\begin{{table}}[{pos}]",
         f"\\caption{{{caption} Source: {src(*sources)}.}}",
         f"\\label{{{label}}}", "\\centering", size]
    if tabcolsep:
        L.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    if tabular == "tabularx":
        L.append(f"\\begin{{tabularx}}{{{width or chr(92) + 'linewidth'}}}{{{colspec}}}")
    else:
        L.append(f"\\begin{{tabular}}{{{colspec}}}")
    L.append("\\toprule")
    hdrs = header if isinstance(header[0], (list, tuple)) else [header]
    for h in hdrs:
        L.append(" & ".join(h) + " \\\\")
    L.append("\\midrule")
    for r in rows:
        if r == "midrule":
            L.append("\\midrule")
        elif isinstance(r, tuple) and r[0] == "group":
            L.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{r[1]}}}}} \\\\")
        else:
            assert len(r) == ncol, (fname, len(r), ncol, r)
            L.append(" & ".join(str(c) for c in r) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabularx}" if tabular == "tabularx" else "\\end{tabular}")
    if notes:
        L.append("\\par\\smallskip")
        L.append("\\begin{minipage}{\\linewidth}\\footnotesize\\raggedright")
        L.append(notes)
        L.append("\\end{minipage}")
    L.append("\\end{table}")
    (OUT / fname).write_text("\n".join(L) + "\n")


def copy_fig(srcpath, dst_name):
    srcpath = Path(srcpath)
    dst = FIGS / dst_name
    shutil.copyfile(srcpath, dst)
    register(srcpath)
    return dst


def crop_png(srcpath, dst_name, top_bp=0.0, bottom_bp=0.0):
    """Physically cropped, alpha-flattened copy of a figure PNG (the original is untouched). Removes in-figure
    titles/footnotes from the file itself, not only from view (a clipped vector PDF would keep the hidden text)."""
    from PIL import Image
    srcpath = Path(srcpath)
    im = Image.open(srcpath)
    dpi = float(im.info.get("dpi", (300, 300))[0])
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg.convert("RGB")
    px = lambda bp: int(round(bp * dpi / 72.0))
    w, h = im.size
    out = im.crop((0, px(top_bp), w, h - px(bottom_bp)))
    dst = FIGS / dst_name
    out.save(dst, dpi=(dpi, dpi))
    register(srcpath)
    return dst


def plot_route3_panel(dst_name):
    """SM-local re-plot of the control-motif figure. The panels are drawn by the experiment's own plotting functions
    (src/upgrades/route3_specificity/make_fig_route3_specificity.py: left_panel, right_panel) from the same two CSVs,
    so the data and marks are identical to results/upgrades/figures/route3_specificity.pdf. Only the legend of panel (b)
    changes: it is moved above the lines it overlapped in the original render, and the split names are spelled out
    ('equalized' for the code name comp_equalized)."""
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    script = ROOT / "src" / "upgrades" / "route3_specificity" / "make_fig_route3_specificity.py"
    spec = importlib.util.spec_from_file_location("r3fig", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # sets the experiment figure's rcParams (8 pt text, TrueType embedding)
    register(script)
    sp = load("route3_specificity.csv")
    ti = load("route3_threshold_invariance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45))
    mod.left_panel(axes[0], sp)
    mod.right_panel(axes[1], ti)
    ax = axes[1]
    h, lab = ax.get_legend_handles_labels()
    lab = [x.replace(" | comp_equalized", ", equalized").replace(" | original", ", original") for x in lab]
    ax.get_legend().remove()
    ax.set_ylim(0.25, 0.68)                  # headroom for the legend; the data range is unchanged
    ax.set_yticks(np.arange(0.25, 0.601, 0.05))   # the original render's ticks
    ax.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=7, handlelength=1.3, handletextpad=0.35,
              borderpad=0.1, borderaxespad=0.1, labelspacing=0.25, columnspacing=0.7)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(FIGS / dst_name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    matplotlib.rcdefaults()
    return FIGS / dst_name


def plot_frozen_probe(dst_name):
    """SM-local vector re-plot of results/upgrades/figures/fm_cross_architecture.png from fm_cross_architecture.csv
    (the rows that figure draws: LightGBM head, GC-matched arm; bars = test effect, whiskers = the file's 95% CI), at print
    size with 8-pt text and plain-language labels. The experiment script (src/upgrades/fm_probe/make_cross_architecture.py)
    is not touched; groups, colours and order follow it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
    ca = load("fm_cross_architecture.csv")
    prim = ca[ca["head"] == "lgbm"]
    groups = [("HyenaDNA-tiny-16k", "sep", "#D55E00", "HyenaDNA [SEP]"),
              ("DNABERT-2-117M", "mean", "#0072B2", "DNABERT-2 mean"),
              ("DNABERT-2-117M", "cls", "#56B4E9", "DNABERT-2 [CLS]")]
    datasets = [("human_enhancers_cohn", "cohn"), ("nt_enhancers", "nt_enhancers"),
                ("drosophila_enhancers_stark", "GC-balanced control\n(drosophila; HyenaDNA only)")]
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    x = np.arange(len(datasets))
    w = 0.25
    n_bars = 0
    for gi, (arch, pool, col, lab) in enumerate(groups):
        ys, lo_, hi_ = [], [], []
        for ds, _ in datasets:
            c = prim[(prim.architecture == arch) & (prim.pooling == pool) & (prim.dataset == ds)]
            assert len(c) <= 1
            if len(c):
                r = c.iloc[0]
                lo, hi = (float(v) for v in str(r["ci"]).strip("[] ").split(","))
                ys.append(r.test_effect_auroc); lo_.append(r.test_effect_auroc - lo); hi_.append(hi - r.test_effect_auroc)
                n_bars += 1
            else:
                ys.append(np.nan); lo_.append(0.0); hi_.append(0.0)
        ax.bar(x + (gi - 1) * w, ys, w, yerr=[lo_, hi_], color=col, label=lab, capsize=2,
               error_kw=dict(elinewidth=0.8, capthick=0.8))
    assert n_bars == 7, n_bars
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in datasets])
    ax.set_ylabel("Test effect (AUROC)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.4, columnspacing=1.2,
              borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / dst_name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    matplotlib.rcdefaults()
    return FIGS / dst_name


def one(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) == 1, (kw, len(q))
    return q.iloc[0]


# ============================================================================ S1
def s1():
    # ---- Pre-specification record ------------------------------------------------------------
    M = ROOT / "src" / "upgrades"
    rows_def = [
        ("Composition matching vs.\\ size-matched random negatives (\\task{psm\\_sensitivity})",
         "Decision rule: criteria (i-a) size, (i-b) separation, (ii-a) overlap weighting, (ii-b) two-sided 1:1 at "
         "caliper 0.2 SD, and the verdict map; primary cell = strictness $\\alpha{=}1$ (the released equalizer "
         "rule), $k$-mer AUROC, retrained within-arm design, selection seed 42. Parts (1)--(6): strictness sweep, "
         "common-support trimming, 10 size-matched random seeds at every level, imbalance-robust metrics, "
         "overlap/ATT weighting with permuted-weight placebo, two-sided 1:1 caliper sweep (2--0.01 SD), "
         "positive-side matching, fixed-model and undistorted-test (train-only) designs, selection bias, "
         "label-noise checks",
         "run\\_psm\\_sensitivity.py (docstring)",
         "P0/P0b duplicate and interval audits; P1 leak-free \\task{nt\\_enhancers} test; P2 leak-free refit; "
         "P3 class-balanced training; P4/P6 between-dataset contrasts; P5 trimming selection bias; P7 "
         "decomposition CIs (extended to the fixed-model design); P8 trimming band check; dissociation statistics",
         [M / "psm_sensitivity" / "run_psm_sensitivity.py", M / "psm_sensitivity" / "run_posthoc.py"]),
        ("Real-task covariates of learnability (\\task{multivar})",
         "Covariate list and transforms; pool A (discriminability plus nine task covariates) and pool B (A plus the composition-only "
         "score, a decomposition that never enters the decision rule); analyses A0--A9; decision rule (a)--(c)",
         "multivar\\_preregistration.txt (frozen docstring)",
         "P1--P13: GC-mean readout, exact permutation $p$ ($n\\le12$), task-identity and 31-mer locus audits, "
         "locus-independent task sets, slope transfer, orientation robustness",
         [RU / "multivar_preregistration.txt", M / "multivar" / "run_multivar.py"]),
        ("End-to-end fine-tuning (\\task{fm\\_finetune})",
         "Contrasts $S{=}A{-}B$, $T{=}C{-}B$, $U{=}E{-}D$, $R{=}(E{-}0.5)/(A{-}0.5)$; rules 1--5 (rule 5: the "
         "GC-balanced control); imbalance-robust metrics; score--composition coupling; written before any v2 unit "
         "finished",
         "eval\\_ft\\_v2.py (docstring)",
         "Share of above-chance AUROC lost; $k$-mer and frozen-probe baselines on identical arms; leak-free "
         "\\task{nt\\_enhancers}; Q1 unit diagnostics, Q2 GC-matched effect, Q3 cCRE sensitivity, Q4 random-draw "
         "reference, Q5 within-class $R^2$, Q6 Bonferroni CIs",
         [M / "fm_finetune" / "eval_ft_v2.py", M / "fm_finetune" / "posthoc_ft_v2.py",
          M / "fm_finetune" / "sensitivity_ft_v2.py"]),
        ("Homology re-clustering (\\task{homology\\_recluster})",
         "MMseqs2 settings (id80 primary, id50 strict); purge vs.\\ class-matched random-removal null; "
         "cluster-aware vs.\\ random re-split; primary Route-1 test (the paper's discriminability vs.\\ id80 "
         "cluster re-split MCC, 11 TF-motif tasks); label-preservation rule; X1--X10 re-runs",
         "run\\_eval.py, run\\_routes.py, run\\_extra.py (docstrings)",
         "Prevalence-adjusted purge MCC; the 0.05-MCC materiality cut; split seeds 43--44 and the seed-robust "
         "call; re-pooling control for relabels; subtype-matched null; audit; containment-coverage sensitivity",
         [M / "homology_recluster" / "run_eval.py", M / "homology_recluster" / "run_routes.py",
          M / "homology_recluster" / "run_extra.py", M / "homology_recluster" / "run_labels_addendum.py"]),
        ("Provenance of the negatives (\\task{neg\\_provenance})",
         "Groups and background draws, features, decision rules D1--D4",
         "run\\_neg\\_provenance.py (docstring)",
         "Protein-coding rule background; D5; train/test split provenance and FASTA-header analysis; six-seed "
         "pooled backgrounds; ChromHMM rule counts",
         [M / "neg_provenance" / "run_neg_provenance.py", M / "neg_provenance" / "nt_split_provenance.py",
          M / "neg_provenance" / "bg_seed_stability.py"]),
        ("Alternative interpretable classifiers (\\task{alt\\_model})",
         "Models and tuning, primary and sensitivity label rules, preservation criterion",
         "run\\_alt\\_model.py (docstring)",
         "Mapping to the paper's Table~I labels; overlap-weighting readout and contrasts; size-controlled "
         "random-negative readouts; EBM variants; uncapped refits (P1--P11)",
         [M / "alt_model" / "run_alt_model.py", M / "alt_model" / "analyze_alt_model.py"]),
        ("Third suite (\\task{gue\\_suite})",
         "Per-task pipeline with the paper's thresholds; predictions P1--P4",
         "run\\_gue.py (docstring)",
         "PH1--PH5, reference run on the paper's 15 tasks, independence-audit checks 4--7",
         [M / "gue_suite" / "run_gue.py", M / "gue_suite" / "emp_provenance.py",
          M / "gue_suite" / "rule_sensitivity.py"]),
    ]
    rows = []
    for mod, pre, where, post, files in rows_def:
        for p in files:
            assert p.exists(), p
            register(p)
        assert mod.count(" (\\task{") == 1, mod
        # the code directory on its own line, so no directory name breaks inside the name
        rows.append([mod.replace(" (\\task{", "\\newline(\\task{"), pre + " \\emph{[" + where + "]}", post])
    pre_txt = read_text(RU / "multivar_preregistration.txt")
    m = re.search(r"sha256 of the 99-line docstring-only file: ([0-9a-f]{64})", pre_txt)
    macro("smMultivarPreregSha", m.group(1)[:12], "multivar_preregistration.txt header line 3")
    write_table(
        "tab_prespec.tex", "tab:prespec",
        "Pre-specification record. `Fixed before' means written in the analysis script (docstring) before the "
        "analysis it governs was run; there is no external registry.",
        ["src/upgrades/*/ (docstrings)", "multivar_preregistration.txt"],
        # first column wide enough for every code-directory name on one line (no break inside homology_recluster)
        ">{\\raggedright\\arraybackslash}p{0.2\\linewidth}>{\\raggedright\\arraybackslash}p{0.42\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.33\\linewidth}",
        ["Analysis (code directory)", "Fixed before the analysis ran \\emph{[where]}", "Added later, labeled post hoc in every output"],
        rows, size="\\footnotesize", tabcolsep="3pt",
        notes="Post hoc analyses are reported as descriptive unless the table says otherwise. Letter--number codes (A0--A9, "
              "D1--D5, P0--P13, PH1--PH5, Q1--Q6, X1--X10) are the identifiers that the analysis scripts, their READMEs and the "
              "result files use; each is described next to it here, and they carry no meaning beyond that description. The "
              "fine-tuning contrasts $S$, $T$, $U$ and $R$ are defined in Section~\\ref{sec:s5ft}; code directories are under "
              "\\task{src/upgrades/}.")

    # ---- Software, data and model versions -----------------------------------------------------
    def ver(pkg, module=None):
        try:
            return im.version(pkg)
        except Exception:
            try:
                return __import__(module or pkg).__version__
            except Exception:
                return "not installed"

    ft_py = ROOT / ".venv-ft" / "bin" / "python"
    ftv = {}
    if ft_py.exists():
        code = ("import importlib.metadata as m\n"
                "for k in ['torch','transformers','peft','tokenizers']:\n"
                "    print(k, m.version(k))")
        outp = subprocess.run([str(ft_py), "-c", code], capture_output=True, text=True, timeout=120).stdout
        for line in outp.splitlines():
            k, v = line.split()
            ftv[k] = v
    try:
        mm = subprocess.run(["mmseqs", "version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        mm = "not found"

    def pin(token, relpath):
        p = ROOT / relpath
        ok = p.exists() and token in p.read_text(errors="ignore")
        assert ok, (token, relpath)
        register(p)
        return pth(relpath)

    srows = [
        ["Python (analysis environment, torch-free)", f"{sys.version.split()[0]}", "interpreter"],
        ["NumPy / pandas / SciPy", f"{np.__version__} / {pd.__version__} / {ver('scipy')}", "installed"],
        ["scikit-learn / LightGBM", f"{ver('scikit-learn')} / {ver('lightgbm')}", "installed"],
        ["interpret (EBM)", ver("interpret-core", "interpret"), "installed"],
        ["minimap2 (mappy)", ver("mappy"), "installed"],
        ["MMseqs2", esc(mm), "\\task{mmseqs version}"],
        ["PyTorch / transformers / PEFT / tokenizers (fine-tuning environment)",
         " / ".join(ftv.get(k, "?") for k in ["torch", "transformers", "peft", "tokenizers"]),
         "\\task{.venv-ft}"],
        ["HyenaDNA-tiny-16k-d128 revision", "\\task{d79fa37e}", pin("d79fa37e", "REPRODUCIBILITY.md")],
        ["DNABERT-2-117M revision", "\\task{7bce263b}", pin("7bce263b", "src/upgrades/fm_finetune/README.md")],
        ["NT downstream tasks, dataset revision", "\\task{96d86d5}", pin("96d86d5", "src/upgrades/neg_provenance/run_neg_provenance.py")],
        ["Genomic Benchmarks archive", "Zenodo record 16605299", pin("16605299", "src/upgrades/homology_recluster/README.md")],
        ["ENCODE SCREEN cCRE registry V4 (GRCh38)", "md5 \\task{60c14d9f}\\ldots", pin("60c14d9f4903a7bd13b3db4f290ace48", "src/upgrades/neg_provenance/README.md")],
        ["Motif database; gene annotation", "JASPAR 2024 CORE; GENCODE v44", pin("GENCODE v44", "src/upgrades/neg_provenance/README.md")],
    ]
    write_table("tab_software.tex", "tab:software",
                "Software, data and model versions (read from the environment when this supplement was generated; "
                "pins verified in the files named).",
                ["REPRODUCIBILITY.md", "code-directory READMEs"],
                ">{\\raggedright\\arraybackslash}p{0.42\\linewidth}>{\\raggedright\\arraybackslash}p{0.28\\linewidth}"
                ">{\\raggedright\\arraybackslash}p{0.24\\linewidth}",
                ["Component", "Version or pin", "Recorded in"], srows, tabcolsep="3pt")

    # ---- D9: DNABERT-2 token lengths of the drosophila control ---------------------------------------
    d9 = RU / "fm_finetune_v2_token_lengths.csv"
    if d9.exists():
        d = load(d9)
        rows = []
        for ds in ["drosophila_enhancers_stark", "human_enhancers_cohn", "nt_enhancers"]:
            for sp in ["train", "test", "all"]:
                q = d[(d.dataset == ds) & (d.split == sp)]
                if not len(q):
                    continue
                r = q.iloc[0]
                rows.append([short(ds) if sp == "train" else "", "train+test" if sp == "all" else sp, intc(r.n),
                             f"{intc(r.len_bp_median)} [{intc(r.len_bp_q1)}, {intc(r.len_bp_q3)}] ({intc(r.len_bp_min)}--{intc(r.len_bp_max)})",
                             f"{intc(r.tokens_median)} [{intc(r.tokens_q1)}, {intc(r.tokens_q3)}] ({intc(r.tokens_min)}--{intc(r.tokens_max)})",
                             f(r.bp_per_nonspecial_token, 2), f"{intc(r.n_over_context)} ({pct(r.frac_over_context, 1)})"])
        r0 = d.iloc[0]
        write_table("tab_d9.tex", "tab:d9",
                    "DNABERT-2 token lengths of the fine-tuning splits, measured with the pinned tokenizer (post hoc, descriptive).",
                    ["fm_finetune_v2_token_lengths.csv"], "llrllrr",
                    ["Data", "Split", "$n$", "Length, bp: median [IQR] (range)", "Tokens: median [IQR] (range)", "bp/token",
                     f"$>${intc(r0.context_tokens)} tokens"], rows,
                    notes="Token counts include the [CLS] and [SEP] tokens and are taken before the fine-tuning code's truncation at 512 "
                          "tokens; bp/token excludes the two special tokens. Tokenizer: \\task{zhihan1996/DNABERT-2-117M} at revision "
                          "\\task{7bce263b} (local snapshot, offline); script \\task{src/upgrades/fm\\_finetune/measure\\_token\\_lengths.py}. "
                          "\\task{drosophila} is the GC-balanced control; \\task{cohn} and \\task{nt\\_enhancers} are shown for context.")
        r = one(d, dataset="drosophila_enhancers_stark", split="all")
        rt = one(d, dataset="drosophila_enhancers_stark", split="train")
        re_ = one(d, dataset="drosophila_enhancers_stark", split="test")
        macro("smDNinePct", pct(r.frac_over_context, 1), "fm_finetune_v2_token_lengths.csv drosophila all frac_over_context")
        macro("smDNineN", f"{intc(r.n_over_context)}/{intc(r.n)}", "same, n_over_context/n")
        macro("smDNineTrain", f"{intc(rt.n_over_context)}/{intc(rt.n)}", "same, train")
        macro("smDNineTest", f"{intc(re_.n_over_context)}/{intc(re_.n)}", "same, test")
        macro("smDNineMedTok", intc(r.tokens_median), "same, tokens_median")
        macro("smDNineStatus", "available", "fm_finetune_v2_token_lengths.csv exists")
    else:
        (OUT / "tab_d9.tex").write_text(
            "% generated: D9 file missing\n\\begin{table}[!htbp]\\caption{DNABERT-2 token lengths of the "
            "fine-tuning splits. Source: \\src{fm\\_finetune\\_v2\\_token\\_lengths.csv}.}"
            "\\label{tab:d9}\\centering\\footnotesize \\fbox{\\parbox{0.8\\linewidth}{\\centering PENDING: "
            "\\task{results/upgrades/fm\\_finetune\\_v2\\_token\\_lengths.csv} was not present when this supplement "
            "was generated.}}\\end{table}\n")
        for n in ["smDNinePct", "smDNineN", "smDNineTrain", "smDNineTest", "smDNineMedTok"]:
            macro(n, "\\textbf{PENDING}", "D9 file missing")
        macro("smDNineStatus", "PENDING", "fm_finetune_v2_token_lengths.csv missing")


# ============================================================================ S2
def s2():
    hc = load("multivar_posthoc_homology_clean.csv")
    dd = load("multivar_posthoc_distinct_datasets.csv")
    li = load("multivar_posthoc_locus_independent.csv")
    links = load("multivar_posthoc_task_locus_links.csv")
    ht = load("homology_recluster_table.csv")
    cov = load("multivar_covariates.csv")
    fam = load("multivar_posthoc_family.csv")

    s11 = one(hc, subset="S11_all")
    c7 = one(hc, subset="rule_clean_homology_lt_0.05")
    p9 = one(hc, subset="paper_homology_clean_9")
    p8 = one(hc, subset="paper9_minus_human_enhancers_ensembl")
    n5 = one(dd, subset="homology_clean_noshare5_drop_regulatory")
    noreg = one(dd, subset="S11_minus_regulatory")
    gb = one(dd, subset="suite_GB_only")
    nt = one(dd, subset="suite_NT_only")
    t25 = li[li.tau == 0.25]
    set5 = set(n5.tasks.split(";"))
    max_link5 = links[links.task_a.isin(set5) & links.task_b.isin(set5)].locus_link.max()
    prom = {"human_nontata_promoters", "nt_promoter_all", "nt_promoter_no_tata", "human_ensembl_regulatory"}
    pl = links[links.task_a.isin(prom) & links.task_b.isin(prom) & (links.exact_shared_frac_max == 0)
               & (links.locus_link >= 0.5)].locus_link
    assert len(pl) == 5, len(pl)
    # every n=7 locus-independent set omits the regulatory task
    assert all("human_ensembl_regulatory" not in s for s in t25.tasks)
    leaky = {"human_nontata_promoters", "human_enhancers_ensembl", "nt_enhancers", "nt_enhancers_types"}
    n_leaky_in_sets = sorted({len(set(s.split(";")) & leaky) for s in t25.tasks})

    def pr(rho, p):
        return f"{f(rho, 2, True)} ({sig2(p)})"

    rows = [
        ["All 11 TF-motif tasks", intc(s11.n), pr(s11.spearman, s11.spearman_p_perm),
         pr(s11.spearman_gc_mcc, s11.spearman_gc_mcc_p_perm), f"$\\rho_\\mathrm{{GC}}-\\rho_\\mathrm{{disc}}$ {ci(s11.rho_gc_minus_rho_disc_lo, s11.rho_gc_minus_rho_disc_hi, 2, True)}"],
        ["Without the 4 tasks with $\\ge$5\\% 8-mer-Jaccard near-duplicates", intc(c7.n), pr(c7.spearman, c7.spearman_p_perm),
         pr(c7.spearman_gc_mcc, c7.spearman_gc_mcc_p_perm), "all below the 5\\% 8-mer threshold"],
        ["\\quad of which the 5 with no exact sequence sharing", intc(n5.n), pr(n5.spearman, n5.spearman_p_perm),
         pr(n5.spearman_gc_mcc, n5.spearman_gc_mcc_p_perm), f"largest locus link {pct(max_link5)}"],
        ["Leaky tasks dropped except \\task{nt\\_enh\\_types}", intc(p8.n), pr(p8.spearman, p8.spearman_p_perm),
         pr(p8.spearman_gc_mcc, p8.spearman_gc_mcc_p_perm), "1 leaky task kept"],
        ["Without \\task{nontata} and \\task{nt\\_enhancers}", intc(p9.n), pr(p9.spearman, p9.spearman_p_perm),
         pr(p9.spearman_gc_mcc, p9.spearman_gc_mcc_p_perm), "keeps 2 leaky tasks"],
        [f"Largest locus-independent sets (link $<$0.25; {len(t25)} sets)", intc(t25.max_unlinked_size.iloc[0]),
         f"{f(t25.spearman_disc_mcc.min(), 2, True)} to {f(t25.spearman_disc_mcc.max(), 2, True)} "
         f"({sig2(t25.spearman_disc_mcc_p_perm.min())}--{sig2(t25.spearman_disc_mcc_p_perm.max())})",
         f"{f(t25.spearman_gc_mcc.min(), 2, True)} ({sig2(t25.spearman_gc_mcc_p_perm.min())})",
         "each omits regulatory"],
        ["Without the regulatory task", intc(noreg.n), pr(noreg.spearman, noreg.spearman_p_perm),
         pr(noreg.spearman_gc_mcc, noreg.spearman_gc_mcc_p_perm), ""],
        ["GB tasks only", intc(gb.n), pr(gb.spearman, gb.spearman_p_perm), pr(gb.spearman_gc_mcc, gb.spearman_gc_mcc_p_perm), ""],
        ["NT tasks only", intc(nt.n), pr(nt.spearman, nt.spearman_p_perm), pr(nt.spearman_gc_mcc, nt.spearman_gc_mcc_p_perm),
         f"{intc(nt.n_exact_distinct_sets)} distinct sequence sets"],
    ]
    assert (t25.spearman_gc_mcc.min() == t25.spearman_gc_mcc.max())
    assert len(set(li[li.tau == 0.25].max_unlinked_size)) == 1 and int(li[li.tau == 0.25].max_unlinked_size.iloc[0]) == 7
    ocr_mm = one(ht, task="human_ocr_ensembl").red_test_id80
    reg_mm = one(ht, task="human_ensembl_regulatory").red_test_id80
    notes = (f"$\\rho$: Spearman correlation of benchmark MCC with the paper's discriminability (disc.) or with the "
             f"class-pooled GC mean (post hoc); $p$: exact permutation. Rows are sensitivity readouts of one correlation, "
             f"not separate tests. The 11 tasks are {intc(s11.n_exact_distinct_sets)} distinct sequence sets once exact "
             f"copies and unions are collapsed; the non-TATA promoter tasks of both suites and the regulatory task's "
             f"promoter class share most loci ({prange(pl.min(), pl.max())}; 31-mer containment). `Near-duplicates' "
             f"uses 8-mer Jaccard $\\ge$0.7; under MMseqs2 id80, \\task{{human\\_ocr\\_ensembl}} ({pct(ocr_mm)}) and the "
             f"regulatory task ({pct(reg_mm)}) also exceed 5\\%. Every largest locus-independent set omits the regulatory "
             f"task and keeps {n_leaky_in_sets[0]} or {n_leaky_in_sets[-1]} of the four leaky tasks.")
    write_table("tab_subsets.tex", "tab:subsets",
                "Route~1 on real tasks: rank correlation of benchmark MCC with discriminability and with GC content "
                "across task subsets.",
                ["multivar_posthoc_homology_clean.csv", "multivar_posthoc_distinct_datasets.csv",
                 "multivar_posthoc_locus_independent.csv", "multivar_posthoc_task_locus_links.csv",
                 "homology_recluster_table.csv"],
                ">{\\raggedright\\arraybackslash}p{0.36\\linewidth}rll>{\\raggedright\\arraybackslash}p{0.2\\linewidth}",
                ["Task set", "$n$", "$\\rho$(disc., MCC) ($p$)", "$\\rho$(GC, MCC) ($p$)", "Note"], rows, notes=notes,
                tabcolsep="4pt")
    macro("smRdiscAll", f(s11.spearman, 2, True), "multivar_posthoc_homology_clean.csv S11_all spearman")
    macro("smPdiscAll", sig2(s11.spearman_p_perm), "same, spearman_p_perm")
    macro("smRgcAll", f(s11.spearman_gc_mcc, 2, True), "same, spearman_gc_mcc")
    macro("smPgcAll", sig2(s11.spearman_gc_mcc_p_perm), "same, spearman_gc_mcc_p_perm")
    macro("smRdiffCI", ci(s11.rho_gc_minus_rho_disc_lo, s11.rho_gc_minus_rho_disc_hi, 2, True), "same, rho_gc_minus_rho_disc_lo/hi")
    macro("smLotoMAE", f(s11.loto_MAE, 2), "same, loto_MAE")
    macro("smLotoMAECI", ci(s11.loto_MAE_lo, s11.loto_MAE_hi, 2), "same, loto_MAE_lo/hi")
    macro("smLotoRsq", f"{f(s11.loto_R2, 2, True)} {ci(s11.loto_R2_lo, s11.loto_R2_hi, 2, True)}", "same, loto_R2 and loto_R2_lo/hi")
    macro("smRdiscGC", f(s11.spearman_disc_gc_mean, 2, True), "same, spearman_disc_gc_mean")
    macro("smNdistinct", intc(s11.n_exact_distinct_sets), "same, n_exact_distinct_sets")
    macro("smLinkRange", prange(pl.min(), pl.max()), "multivar_posthoc_task_locus_links.csv promoter pairs without exact sharing, link >= 0.5 (derived range)")
    macro("smLinkMin", pct(pl.min()), "multivar_posthoc_task_locus_links.csv promoter pairs (derived min)")
    macro("smLinkMax", pct(pl.max()), "same (derived max)")
    macro("smRcleanSeven", f(c7.spearman, 2, True), "homology_clean rule_clean_homology_lt_0.05 spearman")
    macro("smRgcCleanSeven", f(c7.spearman_gc_mcc, 2, True), "same, spearman_gc_mcc")
    macro("smRfive", f(n5.spearman, 2, True), "distinct_datasets homology_clean_noshare5_drop_regulatory spearman")
    macro("smPfive", sig2(n5.spearman_p_perm), "same, spearman_p_perm")
    macro("smRgcFive", f(n5.spearman_gc_mcc, 2, True), "same, spearman_gc_mcc")
    macro("smPgcFive", sig2(n5.spearman_gc_mcc_p_perm), "same, spearman_gc_mcc_p_perm")
    macro("smRsevenLo", f(t25.spearman_disc_mcc.min(), 2, True), "locus_independent tau 0.25 min disc rho")
    macro("smRsevenHi", f(t25.spearman_disc_mcc.max(), 2, True), "same, max")
    macro("smPsevenLo", sig2(t25.spearman_disc_mcc_p_perm.min()), "same, min p")
    macro("smPsevenHi", sig2(t25.spearman_disc_mcc_p_perm.max()), "same, max p")
    macro("smRgcSeven", f(t25.spearman_gc_mcc.min(), 2, True), "same, GC rho")
    macro("smPgcSeven", sig2(t25.spearman_gc_mcc_p_perm.min()), "same, GC p")
    macro("smRnoReg", f(noreg.spearman, 2, True), "distinct_datasets S11_minus_regulatory spearman")
    macro("smRgcNoReg", f(noreg.spearman_gc_mcc, 2, True), "same, spearman_gc_mcc")

    # GC cluster separation
    a5 = one(fam, variant="as_published", analysis="within_promoter_plus_regulatory_n5")
    e6 = one(fam, variant="as_published", analysis="within_enhancer_ocr_n6")
    p4 = one(fam, variant="as_published", analysis="within_promoter_n4")
    macro("smGCsolvedMin", f(a5.gc_min, 3), "multivar_posthoc_family.csv within_promoter_plus_regulatory_n5 gc_min")
    macro("smGCfailMax", f(e6.gc_max, 3), "same file, within_enhancer_ocr_n6 gc_max")
    macro("smGCpromMin", f(p4.gc_min, 3), "same file, within_promoter_n4 gc_min")
    dis = cov.set_index("task")["disc"]
    macro("smDiscTypes", f(dis["nt_enhancers_types"], 3), "multivar_covariates.csv disc nt_enhancers_types")
    macro("smDiscPromTata", f(dis["nt_promoter_tata"], 3), "multivar_covariates.csv disc nt_promoter_tata")

    # ---- covariate models: nested LOTO, reviewer model, noise ----------------------------------------
    nest = load("multivar_nested.csv")
    rev = load("multivar_reviewer_model.csv")
    noise = load("multivar_noise.csv")
    hm = load("multivar_posthoc_homology_models.csv")
    st = load("multivar_posthoc_slope_transfer.csv")
    fam_lbl = {"poolA": "Pool A (disc.\\ + 9 covariates)", "poolB": "Pool B (A + composition-only score)",
               "disc_anchored": "Discriminability + 0--2 pool-A covariates"}
    rows = []
    for cfg, lab in [("S11/Y1_MCC", "11 TF tasks, outcome MCC (primary)"), ("S11/Y2_AUROC", "11 TF tasks, outcome AUROC"),
                     ("S15/Y1_MCC", "all 15 tasks, outcome MCC")]:
        rows.append(("group", lab))
        for _, r in nest[nest.config == cfg].iterrows():
            sel = "AICc" if r.selection == "aicc" else "inner LOTO MAE"
            rows.append([fam_lbl[r.family] + (" $\\dagger$" if r.in_decision_rule else ""), sel,
                         f(r.nested_loto_MAE, 3), f(r.disc_only_loto_MAE, 3),
                         f"{f(r.dMAE, 3, True)} {ci(r.dMAE_lo, r.dMAE_hi, 3, True)}"])
    write_table("tab_nested.tex", "tab:nested",
                "Selection-honest nested leave-one-task-out (LOTO) comparison of covariate models with discriminability "
                "alone.",
                ["multivar_nested.csv"],
                "llrrl", ["Model family", "Selected by", "Nested MAE", "Disc.-only MAE", "$\\Delta$MAE [95\\% CI]"], rows,
                notes="$\\dagger$ the pre-specified decision-rule cell. $\\Delta$MAE $<0$ favors the covariate model; the paired "
                      "task-bootstrap CI uses the same resamples for both models. Pool B contains a performance-derived "
                      "covariate (the composition-only model's score on the same test set), so a pool-B model decomposes "
                      "rather than explains the outcome; it was pre-specified as a decomposition and never enters the decision "
                      "rule.")
    pa = one(nest, config="S11/Y1_MCC", family="poolA", selection="aicc")
    pb = one(nest, config="S11/Y1_MCC", family="poolB", selection="aicc")
    pby2 = one(nest, config="S11/Y2_AUROC", family="poolB", selection="aicc")
    macro("smNestMAE", f(pa.nested_loto_MAE, 3), "multivar_nested.csv S11/Y1_MCC poolA aicc nested_loto_MAE")
    macro("smDiscMAE", f(pa.disc_only_loto_MAE, 3), "same, disc_only_loto_MAE")
    macro("smNestdMAE", f"{f(pa.dMAE, 3, True)} {ci(pa.dMAE_lo, pa.dMAE_hi, 3, True)}", "same, dMAE/_lo/_hi")
    macro("smPoolBMAE", f(pb.nested_loto_MAE, 3), "multivar_nested.csv S11/Y1_MCC poolB aicc")
    macro("smPoolBdMAE", f"{f(pb.dMAE, 3, True)} {ci(pb.dMAE_lo, pb.dMAE_hi, 3, True)}", "same, dMAE")
    macro("smPoolBYtwo", f"{f(pby2.dMAE, 3, True)} {ci(pby2.dMAE_lo, pby2.dMAE_hi, 3, True)}", "multivar_nested.csv S11/Y2_AUROC poolB aicc dMAE")

    rv = rev[rev.config == "S11/Y1_MCC"].iloc[0]
    rv15 = rev[rev.config == "S15/Y1_MCC"].iloc[0]
    hmm = hm[hm.config == "S11/Y1_MCC"].iloc[0]
    ns11 = one(noise, config="S11/Y1_MCC")
    ns15 = one(noise, config="S15/Y1_MCC")
    macro("smRevMAE", f(rv.loto_MAE, 3), "multivar_reviewer_model.csv S11 loto_MAE")
    macro("smRevRsq", f(rv.loto_R2, 2, True), "same, loto_R2")
    macro("smRevMAEmm", f(hmm.reviewer_model_mm80_loto_MAE, 3), "multivar_posthoc_homology_models.csv S11 reviewer_model_mm80_loto_MAE")
    macro("smRevMAEfifteen", f(rv15.loto_MAE, 3), "multivar_reviewer_model.csv S15 loto_MAE")
    macro("smNoiseEleven", pct(ns11.noise_share, 1), "multivar_noise.csv S11 noise_share")
    macro("smNoiseFifteen", pct(ns15.noise_share, 1), "multivar_noise.csv S15 noise_share")
    s = one(st, config="S11/Y1_MCC")
    macro("smSlopeReal", f"{f(s.real_slope, 2, True)} {ci(s.real_slope_boot_lo, s.real_slope_boot_hi, 2, True)}", "multivar_posthoc_slope_transfer.csv S11 real_slope, boot CI")
    macro("smSlopeSyn", f(s.synthetic_slope, 2, True), "same, synthetic_slope")

    # ---- partial correlations -----------------------------------------------------------------
    part = load("multivar_partial.csv")
    lbl = {"len_log": "log mean length", "len_sd_log": "length SD", "gc_mean": "GC mean", "gc_sd": "GC SD within class",
           "gc_gap": "GC class gap $|d|$", "homology": "train/test homology (8-mer Jaccard)", "ntrain_log": "log $n_\\mathrm{train}$",
           "multiclass": "3-class task", "pos_delta": "positional rescue $\\Delta$MCC", "comp_mcc": "composition-only MCC"}
    rows = []
    for _, r in part[part.config == "S11/Y1_MCC"].iterrows():
        rows.append([lbl[r.covariate], f"{f(r.partial_rho, 2, True)} {ci(r.partial_boot_lo, r.partial_boot_hi, 2, True)}",
                     f"{f(r.posthoc_rev_partial_rho_c_given_disc, 2, True)} {ci(r.posthoc_rev_partial_boot_lo, r.posthoc_rev_partial_boot_hi, 2, True)}",
                     sig2(r.posthoc_partial_p_holm)])
    rows.append(["train/test homology (MMseqs2 id80; post hoc)",
                 f"{f(hmm.partial_rho_disc_given_mm80, 2, True)} {ci(hmm.partial_boot_lo, hmm.partial_boot_hi, 2, True)}", "--", "--"])
    write_table("tab_partial.tex", "tab:partial",
                "Partial Spearman correlations on the 11 TF-motif tasks: discriminability given each covariate, and the "
                "covariate given discriminability.",
                ["multivar_partial.csv", "multivar_posthoc_homology_models.csv"],
                "lllr", ["Covariate $c$", "$\\rho$(disc., MCC $|$ $c$) [95\\% CI]", "$\\rho$($c$, MCC $|$ disc.) [95\\% CI]", "Holm $p$"],
                rows, notes=f"Raw $\\rho$(disc., MCC) $=$ {f(part.spearman_disc_y.iloc[0], 2, True)}. CIs: task bootstrap (1000 resamples). "
                            "The reverse partials and the Holm-adjusted $p$ (disc.\\ given $c$, 10 tests; $p$ from $t$ with $n{-}3$ df) are post hoc. With $n{=}11$ tasks "
                            "and collinear covariates, few intervals exclude 0.")
    hp = one(part, config="S11/Y1_MCC", covariate="homology")
    macro("smPartialHom", f"{f(hp.partial_rho, 2, True)} {ci(hp.partial_boot_lo, hp.partial_boot_hi, 2, True)}", "multivar_partial.csv S11 homology")
    macro("smPartialMM", f"{f(hmm.partial_rho_disc_given_mm80, 2, True)} {ci(hmm.partial_boot_lo, hmm.partial_boot_hi, 2, True)}", "multivar_posthoc_homology_models.csv S11")

    # ---- Route 1 under homology control and alternative classifiers --------------------------------------
    r1 = load("homology_recluster_route1.csv")
    am = load("alt_model_route1.csv")
    r1p = r1[r1.x == "published_disc"]
    ylab = {"published_MCC": "original-split MCC (Table~I; reference)", "resplit_id80_MCC": "cluster-aware re-split, seed 42 (pre-specified primary)",
            "resplit_random_MCC": "random re-split, seed 42 (control)", "resplit_id80_seedmean_MCC": "cluster-aware re-split, mean of seeds 42--44",
            "resplit_random_seedmean_MCC": "random re-split, mean of seeds 42--44", "purge_id80_MCC": "homologs purged (id80), raw MCC",
            "purge_id80_stdMCC": "homologs purged (id80), prevalence-adjusted MCC"}
    subs = [("all_TF", "11 tasks"), ("excl_nt_enhancers_types", "10 (no nt\\_enh\\_types)"), ("excl_NT_enhancer_tasks", "9 (no NT enhancer tasks)")]
    rows = []
    for y, yl in ylab.items():
        cells = [yl]
        for sname, _ in subs:
            q = r1p[(r1p.y == y) & (r1p.subset == sname)]
            cells.append(f"{f(q.rho.iloc[0], 2, True)} ({intc(q.n.iloc[0])})" if len(q) else "--")
        rows.append(cells)
    rows.append("midrule")
    for model, mlab in [("lgbm", "LightGBM (paper classifier)"), ("lr", "MCC-tuned L1/L2 LR"), ("ebm", "EBM (128 $k$-mers)")]:
        q = one(am, analysis="route1_disc_vs_mcc_n11_uncapped", model=model)
        rows.append([f"alternative classifier: {mlab}", f"{f(q.rho, 2, True)} ({intc(q.n)}; $p$ {sig2(q.p_perm)})", "--", "--"])
    _pm = set(am[am.analysis == "route1_disc_vs_mcc_n11_uncapped"].p_perm_method.astype(str))
    alt_pmeth = {"monte_carlo": "Monte-Carlo", "exact_enumeration": "exact"}[_pm.pop()] if len(_pm) == 1 else "exact or Monte-Carlo"
    macro("smAltPmethod", alt_pmeth, "alt_model_route1.csv route1_disc_vs_mcc_n11_uncapped p_perm_method")
    write_table("tab_route1_robust.tex", "tab:route1robust",
                "Route~1 real-task correlation $\\rho$(discriminability, MCC) under homology control and under "
                "alternative classifiers; cells give $\\rho$ ($n$).",
                ["homology_recluster_route1.csv", "alt_model_route1.csv"],
                ">{\\raggedright\\arraybackslash}p{0.44\\linewidth}lll",
                ["MCC source", subs[0][1], subs[1][1], subs[2][1]], rows,
                notes="Homology rows are reported as $\\rho$ and $n$ only: the homology analysis reports $t$-approximation $p$-values "
                      f"(for the primary cluster re-split, $t$-approximation $p$ = {sig2(one(r1p, y='resplit_id80_MCC', subset='all_TF').p)}, not "
                      "significant). The purge cannot be scored for \\task{nt\\_enhancers\\_types} (a class is eliminated), so purge "
                      "rows have $n{=}10$; on the original-split MCCs of the same 10 tasks $\\rho=" +
                      f(one(r1p, y='published_MCC', subset='excl_nt_enhancers_types').rho, 2, True) + "$. Alternative-classifier rows: "
                      "uncapped MCCs where available (the EBM keeps its capped MCC on the regulatory task), " + alt_pmeth +
                      " permutation $p$.")
    g = lambda y, s: one(r1p, y=y, subset=s).rho
    macro("smHomPub", f(g("published_MCC", "all_TF"), 2, True), "homology_recluster_route1.csv all_TF published_MCC")
    macro("smHomPrim", f(g("resplit_id80_MCC", "all_TF"), 2, True), "same, resplit_id80_MCC (PRIMARY)")
    macro("smHomPrimP", sig2(one(r1p, y="resplit_id80_MCC", subset="all_TF").p), "same, p (t-approximation)")
    macro("smHomRand", f(g("resplit_random_MCC", "all_TF"), 2, True), "same, resplit_random_MCC")
    macro("smHomPrimMean", f(g("resplit_id80_seedmean_MCC", "all_TF"), 2, True), "same, resplit_id80_seedmean_MCC")
    macro("smHomRandMean", f(g("resplit_random_seedmean_MCC", "all_TF"), 2, True), "same, resplit_random_seedmean_MCC")
    macro("smHomNoNTprim", f(g("resplit_id80_seedmean_MCC", "excl_NT_enhancer_tasks"), 2, True), "same, excl_NT_enhancer_tasks seedmean cluster")
    macro("smHomNoNTrand", f(g("resplit_random_seedmean_MCC", "excl_NT_enhancer_tasks"), 2, True), "same, excl_NT seedmean random")
    macro("smHomNoNTpub", f(g("published_MCC", "excl_NT_enhancer_tasks"), 2, True), "same, excl_NT published")
    macro("smHomNoNTsFortyTwo", f(g("resplit_id80_MCC", "excl_NT_enhancer_tasks"), 2, True), "same, excl_NT resplit seed 42")
    macro("smHomPurge", f(g("purge_id80_MCC", "all_TF"), 2, True), "same, purge_id80_MCC (n=10)")
    macro("smHomPurgePub", f(g("published_MCC", "excl_nt_enhancers_types"), 2, True), "same, excl_nt_enhancers_types published")
    for model, nm in [("lr", "LR"), ("ebm", "EBM")]:
        q = one(am, analysis="route1_disc_vs_mcc_n11_uncapped", model=model)
        macro(f"smAltRone{nm}", f(q.rho, 2, True), f"alt_model_route1.csv route1_disc_vs_mcc_n11_uncapped {model}")
        macro(f"smAltRone{nm}P", sig2(q.p_perm), f"same, {model} p_perm")

    crop_png(RU / "figures" / "multivar_covariates.png", "multivar_covariates_cropped.png", top_bp=0, bottom_bp=31)


# ============================================================================ S3
def s3():
    # ---- TATA-box rates: all sequences (this section) vs. the seeded 4,000-per-class sample of the motif scan and the gate
    ca_c = load("contamination_analysis.csv", base=RR)
    ca_n = load(RR / "nt" / "contamination_analysis.csv")
    tb_c = one(ca_c[ca_c.measure == "TATA_box_fraction(>=1)"].drop_duplicates(), dataset="human_enhancers_cohn")
    tb_n = one(ca_n[ca_n.measure == "TATA_box_fraction(>=1)"].drop_duplicates(), dataset="nt_enhancers")
    assert int(tb_c.n_pos) == int(tb_c.n_neg) == int(tb_n.n_pos) == int(tb_n.n_neg)
    smp_all = load("neg_provenance_summary.csv")
    whl = load("neg_provenance_nt_wholeclass.csv")
    a_cp = one(smp_all, dataset="cohn", group="POS", metric="tata_rate").value
    a_cn = one(smp_all, dataset="cohn", group="NEG", metric="tata_rate").value
    a_w = one(whl, split="all", metric="tata_rate")
    macro("smTataSampleN", intc(tb_c.n_pos), "results/contamination_analysis.csv TATA_box_fraction n_pos (per class)")
    macro("smTataSubCohn", f"{pct(tb_c.pos_mean, 1)} vs.\\ {pct(tb_c.neg_mean, 1)} ({f(100 * (tb_c.neg_mean - tb_c.pos_mean), 1, True)} points)",
          "results/contamination_analysis.csv cohn TATA_box_fraction(>=1) pos_mean/neg_mean (4,000 per class)")
    macro("smTataSubNt", f"{pct(tb_n.pos_mean, 1)} vs.\\ {pct(tb_n.neg_mean, 1)} ({f(100 * (tb_n.neg_mean - tb_n.pos_mean), 1, True)})",
          "results/nt/contamination_analysis.csv nt_enhancers TATA_box_fraction(>=1) (4,000 per class)")
    macro("smTataAllCohn", f"{pct(a_cp, 1)} vs.\\ {pct(a_cn, 1)} ({f(100 * (a_cn - a_cp), 1, True)} points)",
          "neg_provenance_summary.csv cohn POS/NEG tata_rate (all sequences)")
    macro("smTataAllNt", f"{pct(a_w.pos, 1)} vs.\\ {pct(a_w.neg, 1)} ({f(100 * (a_w.neg - a_w.pos), 1, True)})",
          "neg_provenance_nt_wholeclass.csv all tata_rate pos/neg (whole class, all sequences)")
    # ---- coordinate (TSS) analysis: scope of the leakage refutation (cohn: all negatives; nt_enhancers: genomic subset only)
    xco = one(load("homology_recluster_extra_coords.csv"), task="human_enhancers_cohn", data="original")
    ntc = one(load("nt_coords.csv"), dataset="nt_enhancers")
    macro("smCohnTssP", sig2(xco.fisher_p), "homology_recluster_extra_coords.csv cohn original fisher_p (TATA-flagged vs unflagged negatives at a TSS)")
    macro("smNtTssP", sig2(ntc.fisher_p), "nt_coords.csv nt_enhancers fisher_p (genomic subset)")
    macro("smNtMapped", pct(ntc.mapped_frac, 1), "nt_coords.csv nt_enhancers mapped_frac")
    tata_note = (f"TATA-box rates here use every sequence of each group. The TATA gate and the motif-only scans "
                 f"(Table~\\ref{{tab:ctlmotif}}) use a seeded sample of {intc(tb_c.n_pos)} sequences per class, which gives "
                 f"{pct(tb_c.pos_mean, 1)} vs.\\ {pct(tb_c.neg_mean, 1)} ({f(100 * (tb_c.neg_mean - tb_c.pos_mean), 1, True)} points) for "
                 f"\\task{{cohn}} positives vs.\\ negatives and {pct(tb_n.pos_mean, 1)} vs.\\ {pct(tb_n.neg_mean, 1)} "
                 f"({f(100 * (tb_n.neg_mean - tb_n.pos_mean), 1, True)}) for the whole \\task{{nt\\_enhancers}} class; on all sequences the "
                 f"excess is {f(100 * (a_cn - a_cp), 1, True)} and {f(100 * (a_w.neg - a_w.pos), 1, True)} points "
                 f"(\\task{{nt\\_enhancers}} whole class: {pct(a_w.pos, 1)} vs.\\ {pct(a_w.neg, 1)}; Table~\\ref{{tab:provd1}}).")

    ct = load("neg_provenance_compact_table.csv")
    lead = lambda s: s if not isinstance(s, str) else s.split(" [")[0]
    cols = ["GC (window)", "TATA-box rate", "AT-rich isochore (L1+L2)", ">=15 kb from protein-coding gene",
            "any ENCODE cCRE overlap", "ChromHMM state 13 (low signal) in all 9 lines", "enhancer state in >=1 line"]
    for c in cols:
        assert c in ct.columns, c
    rows = []
    for _, r in ct.iterrows():
        g = str(r.group).replace("pre-registered", "pre-specified")
        cells = [esc(g).replace(">=", "$\\ge$"), intc(r.n)]
        for c in cols:
            v = lead(r[c])
            cells.append("n/a" if (isinstance(v, str) and v.startswith("n/a")) else (esc(v) if isinstance(v, str) else f(v, 3)))
        rows.append(cells)
    write_table("tab_prov_compact.tex", "tab:provcompact",
                "Provenance of the negatives: composition and genomic context of positives, negatives and rule-drawn or random "
                "genomic backgrounds.",
                ["neg_provenance_compact_table.csv"],
                ">{\\raggedright\\arraybackslash}p{0.27\\linewidth}rrrrrrrr",
                ["Group", "$n$", "GC", "TATA", "L1+L2", "$\\ge$15\\,kb pc", "cCRE", "S13 all 9", "enh.\\ $\\ge$1"],
                rows, tabcolsep="3.5pt",
                notes="Point estimates; the 95\\% bootstrap CIs (window = resampling unit) are in the source file. GC: local window GC; "
                      "TATA: fraction with a TBP hit $\\ge$1.3 bits/pos; L1+L2: 100-kb GC $<$41\\%; $\\ge$15\\,kb pc: distance "
                      "$\\ge$15\\,kb from every GENCODE v44 protein-coding gene; cCRE: any ENCODE V4 cCRE overlap; S13 all 9: "
                      "ChromHMM low-signal state 13 (a nine-mark model without H3K9me3) in all nine ENCODE "
                      "lines; enh.\\ $\\ge$1: an enhancer state in at least one line. `Rule' rows are backgrounds re-drawn by the "
                      "documented sampling rules; which rule variant is pre-specified and which was added post hoc is stated in "
                      "each row. RankGAN-synthetic sequences have no locus (n/a). " + tata_note)

    tests = load("neg_provenance_tests.csv")
    whole = load("neg_provenance_nt_wholeclass.csv")
    d1 = tests[tests.test == "D1_decomposition"]
    mlab = {"gc_local": "GC (window)", "gc_100kb": "GC (100 kb)", "tata_rate": "TATA-box rate", "iso_AT_L1L2": "L1+L2 isochore"}
    rows = []
    for ds, dl in [("cohn", "\\task{cohn}"), ("nt_enhancers", "genomic \\task{nt\\_enh.}")]:
        for mkey in ["gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2"]:
            r = one(d1, dataset=ds, metric=mkey)
            rows.append([dl, mlab[mkey], f(r.gap_pos_minus_neg, 3, True), f(r.bguni_minus_neg, 3, True),
                         f"{pct(r.share_of_gap_from_neg_side)} {pci(r.share_of_gap_from_neg_side_ci_lo, r.share_of_gap_from_neg_side_ci_hi)}",
                         f"{f(r.f_rule, 2)} {ci(r.f_rule_ci_lo, r.f_rule_ci_hi, 2)}", esc(str(r.verdict).replace('_', ' ').lower())])
    for mkey in ["gc_local", "tata_rate"]:
        r = one(whole, split="all", metric=mkey)
        rows.append(["whole \\task{nt\\_enh.} class$^\\ast$", mlab[mkey], f(r.pos - r.neg, 3, True), f(r.bg_uni - r.neg, 3, True),
                     f"{pct(r.share_of_gap_from_neg_side, 1)} {pci(r.ci_lo, r.ci_hi)}", "--", "--"])
    write_table("tab_prov_d1.tex", "tab:provd1",
                "How much of the positive--negative composition gap the negatives' sampling explains (decision rule D1).",
                ["neg_provenance_tests.csv", "neg_provenance_nt_wholeclass.csv"],
                "llrrlll",
                ["Data", "Metric", "POS$-$NEG", "Random$-$NEG", "Neg.-side share [CI]", "$f_\\mathrm{rule}$ [CI]", "D1 verdict"],
                rows, tabcolsep="3pt", size="\\scriptsize",
                notes="$^\\ast$ post hoc; mostly RankGAN-synthetic sequences. The gap POS$-$NEG splits into (POS$-$random genome) $+$ (random genome$-$NEG); the negative-side share is the second "
                      "term's share: the part of the gap that comes from where the negatives were sampled. The rest is the positives' "
                      "departure from random genome (their GC-richness and TATA depletion). $f_\\mathrm{rule}$ = (random$-$rule "
                      "background)/(random$-$NEG) for the pre-specified rule backgrounds (\\task{cohn}: $\\ge$15\\,kb from any GENCODE gene; "
                      "\\task{nt\\_enhancers}: literal non-enhancer states in all nine lines); $f_\\mathrm{rule}{>}1$ means the rule "
                      "background over-shoots the negatives' AT shift. The verdict answers only whether that documented rule reproduces the "
                      "negatives' shift from random genome. All sequences are used (for the seeded 4,000-per-class sample of the motif scan, "
                      "see the note to Table~\\ref{tab:provcompact}).")
    for ds, nm in [("cohn", "Cohn"), ("nt_enhancers", "Nt")]:
        for mkey, mn in [("gc_local", "GC"), ("tata_rate", "Tata")]:
            r = one(d1, dataset=ds, metric=mkey)
            macro(f"smShare{nm}{mn}", pct(r.share_of_gap_from_neg_side), f"neg_provenance_tests.csv D1 {ds} {mkey} share_of_gap_from_neg_side")
            macro(f"smPosShare{nm}{mn}", pct(1 - r.share_of_gap_from_neg_side), "derived: 1 - share")
            macro(f"smShare{nm}{mn}CI", pci(r.share_of_gap_from_neg_side_ci_lo, r.share_of_gap_from_neg_side_ci_hi), "same, CI")
    rw = one(whole, split="all", metric="gc_local")
    macro("smShareWholeGC", pct(rw.share_of_gap_from_neg_side, 1), "neg_provenance_nt_wholeclass.csv all gc_local")
    macro("smPosShareWholeGC", pct(1 - rw.share_of_gap_from_neg_side), "derived")
    rw2 = one(whole, split="all", metric="tata_rate")
    macro("smShareWholeTata", pct(rw2.share_of_gap_from_neg_side, 1), "neg_provenance_nt_wholeclass.csv all tata_rate")
    for ds, nm in [("cohn", "Cohn"), ("nt_enhancers", "Nt")]:
        r = one(d1, dataset=ds, metric="tata_rate")
        macro(f"smTataShift{nm}", f(-100 * r.bguni_minus_neg, 1, True), f"neg_provenance_tests.csv D1 {ds} tata_rate -bguni_minus_neg (points)")

    # post hoc rule backgrounds (pooled six seeds)
    bp = load("neg_provenance_bg_pooled.csv")
    for ds, grp, mkey, nm in [("cohn", "BG_rule_pc", "gc_local", "PcGC"), ("cohn", "BG_rule_pc", "tata_rate", "PcTata"),
                              ("nt_enhancers", "BG_rule13", "gc_local", "SthirteenGC"), ("nt_enhancers", "BG_rule13", "tata_rate", "SthirteenTata")]:
        r = one(bp, dataset=ds, group=grp, metric=mkey)
        macro(f"smBg{nm}", f"{f(r.bg_minus_neg, 4, True)} {ci(r.bg_minus_neg_lo, r.bg_minus_neg_hi, 4, True)}",
              f"neg_provenance_bg_pooled.csv {ds} {grp} {mkey} bg_minus_neg")

    # cohn compartments
    sm = load("neg_provenance_summary.csv")
    gsm = lambda g, m: one(sm, dataset="cohn", group=g, metric=m)
    macro("smCohnPc", pct(gsm("NEG", "ge15kb_gene_pc").value, 1), "neg_provenance_summary.csv cohn NEG ge15kb_gene_pc")
    macro("smCohnPcCI", pci(gsm("NEG", "ge15kb_gene_pc").ci_lo, gsm("NEG", "ge15kb_gene_pc").ci_hi, 1), "same, CI")
    macro("smRandPc", pct(gsm("BG_uni", "ge15kb_gene_pc").value, 1), "neg_provenance_summary.csv cohn BG_uni ge15kb_gene_pc")
    macro("smCohnAll", pct(gsm("NEG", "ge15kb_gene_all").value, 1), "neg_provenance_summary.csv cohn NEG ge15kb_gene_all")
    macro("smCohnIntron", pct(gsm("NEG", "comp_intron").value, 1), "neg_provenance_summary.csv cohn NEG comp_intron")
    macro("smCohnExon", pct(gsm("NEG", "comp_exon").value, 1), "neg_provenance_summary.csv cohn NEG comp_exon")
    macro("smCohnInGene", pct(gsm("NEG", "comp_intron").value + gsm("NEG", "comp_exon").value, 1), "derived: comp_intron + comp_exon")
    macro("smCohnInter", pct(gsm("NEG", "comp_intergenic").value, 1), "neg_provenance_summary.csv cohn NEG comp_intergenic")
    macro("smCohnSthirteen", pct(gsm("NEG", "hmm_13_all9").value, 1), "neg_provenance_summary.csv cohn NEG hmm_13_all9")
    macro("smCohnEnhAny", pct(gsm("NEG", "hmm_enh_any").value, 1), "neg_provenance_summary.csv cohn NEG hmm_enh_any")
    macro("smCohnIso", pct(gsm("NEG", "iso_AT_L1L2").value, 1), "neg_provenance_summary.csv cohn NEG iso_AT_L1L2")
    macro("smRandIso", pct(gsm("BG_uni", "iso_AT_L1L2").value, 1), "neg_provenance_summary.csv cohn BG_uni iso_AT_L1L2")
    ntsm = lambda g, m: one(sm, dataset="nt_enhancers", group=g, metric=m)
    macro("smNtIso", pct(ntsm("NEG", "iso_AT_L1L2").value, 1), "neg_provenance_summary.csv nt_enhancers NEG iso_AT_L1L2")
    macro("smNtRandIso", pct(ntsm("BG_uni", "iso_AT_L1L2").value, 1), "neg_provenance_summary.csv nt_enhancers BG_uni iso_AT_L1L2")

    # nt by split
    bs = load("neg_provenance_nt_by_split.csv")
    org = load("neg_provenance_nt_origin.csv")
    mm = [("hmm_13_all9", "ChromHMM state 13 in all 9 lines"), ("hmm_enh_any", "enhancer state in $\\ge$1 line"),
          ("gc_local", "GC (window)"), ("tata_rate", "TATA-box rate"), ("iso_AT_L1L2", "L1+L2 isochore"), ("ccre_any", "any cCRE overlap")]
    rows = []
    for mkey, ml in mm:
        tr = one(bs, split="train", group="NEG", metric=mkey)
        te = one(bs, split="test", group="NEG", metric=mkey)
        po = one(bs, split="train", group="POS", metric=mkey)
        rows.append([ml, f"{f(tr.value, 3)} {ci(tr.ci_lo, tr.ci_hi, 3)}", f"{f(te.value, 3)} {ci(te.ci_lo, te.ci_hi, 3)}", f(po.value, 3)])
    ntr = int(one(bs, split="train", group="NEG", metric="hmm_13_all9").n)
    nte = int(one(bs, split="test", group="NEG", metric="hmm_13_all9").n)
    orow = lambda s, l, o: one(org, split=s, label=l, origin=o)
    syn_tr_neg = int(orow("train", 0, "synthetic").n_rows)
    gen_tr_neg = int(orow("train", 0, "iEnhancer_original").n_rows)
    syn_tr_pos = int(orow("train", 1, "synthetic").n_rows)
    gen_tr_pos = int(orow("train", 1, "iEnhancer_original").n_rows)
    all_neg = int(orow("all", 0, "synthetic").n_rows + orow("all", 0, "iEnhancer_original").n_rows)
    all_pos = int(orow("all", 1, "synthetic").n_rows + orow("all", 1, "iEnhancer_original").n_rows)
    n_syn_all = int(orow("all", 0, "synthetic").n_rows + orow("all", 1, "synthetic").n_rows)
    n_all = all_neg + all_pos
    write_table("tab_prov_split.tex", "tab:provsplit",
                "Genomic \\task{nt\\_enhancers} negatives by split: training (iEnhancer-2L) vs.\\ test (iEnhancer-EL independent set).",
                ["neg_provenance_nt_by_split.csv", "neg_provenance_nt_origin.csv"],
                "lllr", ["Metric", f"Train negatives ($n{{=}}{intc(ntr)}$) [CI]", f"Test negatives ($n{{=}}{intc(nte)}$) [CI]", "Train positives"],
                rows, notes=(f"Synthetic share (RankGAN; no locus): {intc(syn_tr_neg)} of {intc(syn_tr_neg + gen_tr_neg)} training negatives "
                             f"({pct(syn_tr_neg / (syn_tr_neg + gen_tr_neg), 1)}), {intc(syn_tr_pos)} of {intc(syn_tr_pos + gen_tr_pos)} training positives "
                             f"({pct(syn_tr_pos / (syn_tr_pos + gen_tr_pos), 1)}); {pct(orow('all', 0, 'synthetic').frac_of_class, 1)} of all "
                             f"{intc(all_neg)} negatives and {pct(n_syn_all / n_all, 1)} of all {intc(n_all)} sequences; no test sequence is synthetic. "
                             "The test negatives' FASTA headers point to single ChromHMM state-13 segments rather than to the nine-line intersection "
                             "(inferred post hoc from the headers)."))
    macro("smSynTrNeg", pct(syn_tr_neg / (syn_tr_neg + gen_tr_neg), 1), "derived from neg_provenance_nt_origin.csv train label 0 n_rows")
    macro("smSynTrNegN", f"{intc(syn_tr_neg)}/{intc(syn_tr_neg + gen_tr_neg)}", "same")
    macro("smSynTrPos", pct(syn_tr_pos / (syn_tr_pos + gen_tr_pos), 1), "derived from neg_provenance_nt_origin.csv train label 1 n_rows")
    macro("smSynAllNeg", pct(orow("all", 0, "synthetic").frac_of_class, 1), "neg_provenance_nt_origin.csv all label 0 synthetic frac_of_class")
    macro("smGenomicAll", pct(1 - n_syn_all / n_all, 1), "derived: genomic share of all sequences")
    macro("smGenomicAllN", f"{intc(n_all - n_syn_all)}/{intc(n_all)}", "derived")
    s13tr = one(bs, split="train", group="NEG", metric="hmm_13_all9")
    s13te = one(bs, split="test", group="NEG", metric="hmm_13_all9")
    macro("smStrainNeg", pct(s13tr.value, 1), "neg_provenance_nt_by_split.csv train NEG hmm_13_all9")
    macro("smStestNeg", pct(s13te.value, 1), "neg_provenance_nt_by_split.csv test NEG hmm_13_all9")
    macro("smStestNegCI", pci(s13te.ci_lo, s13te.ci_hi, 1), "same, CI")
    macro("smSrand", pct(one(sm, dataset="nt_enhancers", group="BG_uni", metric="hmm_13_all9").value, 1), "neg_provenance_summary.csv nt BG_uni hmm_13_all9")

    # ChromHMM rule counts and chromosome tests
    rc = load("neg_provenance_chromhmm_rule_counts.csv")
    rows = [[esc(str(r.rule).replace("Heterochrom/lo", "low signal")), intc(r.n_200bp_fragments), intc(r.published_count), f(r.ratio_fragments_to_published, 4)] for _, r in rc.iterrows()]
    write_table("tab_prov_rules.tex", "tab:provrules",
                "Which ChromHMM definition reproduces the published iEnhancer-2L candidate counts (post hoc).",
                ["neg_provenance_chromhmm_rule_counts.csv"], "lrrr",
                ["Rule (hg19 ChromHMM, nine ENCODE lines)", "200-bp fragments", "Published count", "Ratio"], rows,
                notes="Published counts: non-enhancer candidates for the non-enhancer rules, 742 strong enhancers for the strong rule, "
                      "weak-enhancer candidates for the weak rules. `State 13 in all nine' reproduces the non-enhancer count; it does not "
                      "separate state 13 from states 13--15, and no reading reproduces the weak count.")
    r13 = rc[rc.rule.str.startswith("state13_all9")].iloc[0]
    macro("smRuleCount", intc(r13.n_200bp_fragments), "neg_provenance_chromhmm_rule_counts.csv state13_all9 n_200bp_fragments")
    macro("smRulePub", intc(r13.published_count), "same, published_count")
    chrt = load("neg_provenance_chrom_tests.csv")
    chrom = load("neg_provenance_chrom.csv")
    ntc = one(chrt, dataset="nt_enhancers", reference="BG_rule13", method="homogeneity")
    coc = one(chrt, dataset="cohn", reference="BG_rule_pc", method="homogeneity")
    macro("smChromNtChi", f"$\\chi^2={f(ntc.stat, 1)}$, df {int(ntc.df)}, $p$ = {sig2(ntc.p)}", "neg_provenance_chrom_tests.csv nt BG_rule13 homogeneity")
    macro("smChromCohnP", f"$p<10^{{{int(math.ceil(coc.log10_p))}}}$", "neg_provenance_chrom_tests.csv cohn BG_rule_pc homogeneity log10_p (ceil)")
    c9 = one(chrom, dataset="cohn", chr="chr9")
    macro("smChrNineNeg", intc(c9.NEG), "neg_provenance_chrom.csv cohn chr9 NEG")
    macro("smChrNinePos", intc(c9.n_POS), "same, n_POS")
    macro("smChrNineExp", intc(c9.exp_match), "same, exp_match (BG_rule_pc)")


# ============================================================================ S4
def s4():
    dec = load("psm_sensitivity_decision.csv").set_index("dataset")
    summ = load("psm_sensitivity_summary.csv")
    cal = load("psm_sensitivity_caliper_sweep.csv")
    dci = load("psm_sensitivity_decomposition_ci.csv")
    bet = load("psm_sensitivity_between_dataset.csv")
    sel = load("psm_sensitivity_selection_bias.csv")
    dis = load("psm_sensitivity_dissociation_stats.csv")
    lk = load("psm_sensitivity_leakage_check.csv")
    e1 = load("homology_recluster_e1_draws_summary.csv")
    C, N, NL, NR, D = ("human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfreeTest", "nt_enhancers_leakfreeRefit",
                       "drosophila_enhancers_stark")
    DSL = {C: "\\task{cohn}", N: "\\task{nt\\_enhancers} (uncorrected)", NL: "\\task{nt\\_enhancers}, leak-free test",
           NR: "\\task{nt\\_enhancers}, leak-free refit", D: "GC-balanced control"}

    # ---- decision table. The rule and the cohn / original nt_enhancers / control rows are pre-specified; the two
    # leak-free nt_enhancers rows apply the same rule post hoc (P1/P2; their role column says so).
    lk_ = one(lk, dataset="nt_enhancers")
    assert "post hoc" in str(dec.loc[NL].role) and "post hoc" in str(dec.loc[NR].role) and "post hoc" not in str(dec.loc[N].role)
    assert dec.loc[N].verdict == dec.loc[NL].verdict == dec.loc[NR].verdict == "NOT SUPPORTED"
    DSL_DEC = {C: DSL[C], N: "\\task{nt\\_enhancers}, original test", NL: "\\task{nt\\_enhancers}, leak-free test$^\\dagger$",
               NR: "\\task{nt\\_enhancers}, leak-free refit$^\\dagger$", D: DSL[D]}
    rows = []
    for ds in [C, N, NL, NR, D]:
        r = dec.loc[ds]
        rows.append([DSL_DEC[ds], f"{f(r.A0)} {ci(r.A0_lo, r.A0_hi)}", f"{f(r.Ar_mean)} ({f(r.Ar_min)}--{f(r.Ar_max)})",
                     f"{f(r.Am)} {ci(r.Am_lo, r.Am_hi)}", f"{f(r.Am_seedmin)}--{f(r.Am_seedmax)}", f(r.paper_anchor_auroc),
                     pct(r.retained_neg_frac, 1), intc(r.f1_n_test_neg), f(r.f1_max_abs_smd, 2),
                     ", ".join(n for c, n in [("crit_ia_size", "(i-a)"), ("crit_ib_separation", "(i-b)"), ("crit_iia_weighting", "(ii-a)"),
                                              ("crit_iib_twosided", "(ii-b)")] if not bool(r[c])) or "none",
                     esc(str(r.verdict).lower().replace("supported-with-size-component", "supported, with a size component"))])
    write_table("tab_psm_decision.tex", "tab:psmdecision",
                "Decision table for composition matching at the released equalizer's retention ($\\alpha{=}1$; $k$-mer AUROC; "
                "retrained within-arm design). The rule and the \\task{cohn}, original \\task{nt\\_enhancers} and control rows are "
                "pre-specified; the two leak-free \\task{nt\\_enhancers} rows apply the same rule post hoc.",
                ["psm_sensitivity_decision.csv"],
                ">{\\raggedright\\arraybackslash}p{0.11\\linewidth}lllllrrr>{\\raggedright\\arraybackslash}p{0.055\\linewidth}>{\\raggedright\\arraybackslash}p{0.085\\linewidth}",
                ["Data", "Original [CI]", "Random (10 draws)", "Matched, seed 42 [CI]", "Seeds 42--44", "Released", "Kept", "Test neg.",
                 "max$|$SMD$|$", "Failed", "Verdict"], rows, size="\\scriptsize", tabcolsep="2pt",
                notes="Released: AUROC on the released equalized split. Random: mean (range) over 10 size-matched random negative draws; matched: the released equalizer rule re-drawn with "
                      "selection seed 42 (the verdict uses seed 42, as pre-specified). Criteria: (i-a) size effect $\\le$ tolerance "
                      "$\\tau=\\max(0.03$, half-width of the original CI); (i-b) matched below every random seed and its upper CI below the "
                      "random mean; (ii-a) overlap-weighted AUROC below every permuted-weight placebo, upper CI below the placebo mean, and "
                      "a drop $\\ge$0.5$\\times$ the matching drop; (ii-b) the same for two-sided 1:1 matching at caliper 0.2 SD. SUPPORTED: "
                      "all four; SUPPORTED-WITH-SIZE-COMPONENT: all but (i-a); NOT SUPPORTED: (i-b) fails or both (ii) fail. max$|$SMD$|$ of "
                      "the re-drawn matched set; the released equalized \\task{cohn} split has " + f(dec.loc[C].paper_anchor_max_abs_smd, 2) +
                      f". {intc(lk_.n_test_verbatim_in_train)} of the {intc(lk_.n_test)} \\task{{nt\\_enhancers}} test sequences, all of them positives, "
                      "occur verbatim in the training split. $^\\dagger$Post hoc: the leak-free rows remove them from every test set (leak-free test; "
                      "same models) or refit every arm without their training copies and score the full test set (leak-free refit). "
                      "The verdict is the same in all three \\task{nt\\_enhancers} rows.")
    rc = dec.loc[C]
    macro("smAzero", f(rc.A0), "psm_sensitivity_decision.csv cohn A0")
    macro("smAzeroCI", ci(rc.A0_lo, rc.A0_hi), "same, A0_lo/hi")
    macro("smAr", f(rc.Ar_mean), "psm_sensitivity_decision.csv cohn Ar_mean")
    macro("smArRange", f"{f(rc.Ar_min)}--{f(rc.Ar_max)}", "same, Ar_min/max")
    macro("smAm", f(rc.Am), "psm_sensitivity_decision.csv cohn Am (seed 42)")
    macro("smAmCI", ci(rc.Am_lo, rc.Am_hi), "same, Am_lo/hi")
    macro("smAmSeeds", f"{f(rc.Am_seedmin)}--{f(rc.Am_seedmax)}", "same, Am_seedmin/max")
    macro("smPub", f(rc.paper_anchor_auroc), "psm_sensitivity_decision.csv cohn paper_anchor_auroc")
    macro("smKeptCohn", pct(rc.retained_neg_frac, 1), "psm_sensitivity_decision.csv cohn retained_neg_frac")
    macro("smSMDpub", f(rc.paper_anchor_max_abs_smd, 2), "psm_sensitivity_decision.csv cohn paper_anchor_max_abs_smd")
    macro("smSMDredraw", f(rc.f1_max_abs_smd, 2), "psm_sensitivity_decision.csv cohn f1_max_abs_smd")
    macro("smSMDntPub", f(dec.loc[N].paper_anchor_max_abs_smd, 2), "psm_sensitivity_decision.csv nt paper_anchor_max_abs_smd")
    macro("smNtTestNeg", intc(dec.loc[NL].f1_n_test_neg), "psm_sensitivity_decision.csv nt leak-free f1_n_test_neg")
    macro("smSizeEff", f"{f(rc.p7_size_effect, 3, True)} {ci(rc.p7_size_effect_lo, rc.p7_size_effect_hi, 3, True)}", "decision.csv cohn p7_size_effect")
    macro("smSizeShare", pct(rc.p7_size_share_of_total), "decision.csv cohn p7_size_share_of_total")
    macro("smSizeShareCI", pci(rc.p7_size_share_of_total_lo, rc.p7_size_share_of_total_hi), "same, CI")
    macro("smMvr", f"{f(rc.p7_matched_vs_random, 3, True)} {ci(rc.p7_matched_vs_random_lo, rc.p7_matched_vs_random_hi, 3, True)}", "decision.csv cohn p7_matched_vs_random")
    macro("smMvrShare", pct(rc.p7_mvr_share_of_total), "decision.csv cohn p7_mvr_share_of_total")
    macro("smMvrShareCI", pci(rc.p7_mvr_share_of_total_lo, rc.p7_mvr_share_of_total_hi), "same, CI")
    macro("smDropTotal", f(rc.drop_total, 3), "decision.csv cohn drop_total")
    macro("smMvrPoint", f(rc.p7_matched_vs_random, 3), "decision.csv cohn p7_matched_vs_random")
    macro("smAzeroNt", f(dec.loc[N].A0), "decision.csv nt_enhancers A0")
    macro("smCompZeroNt", f(dec.loc[N].comp0), "decision.csv nt_enhancers comp0")
    s4244 = rc.p7_s4244_size_effect / (rc.p7_s4244_size_effect + rc.p7_s4244_matched_vs_random)
    macro("smSizeShareSeeds", pct(s4244), "derived: p7_s4244_size_effect/(size+matched_vs_random), seeds 42-44")
    # size share at the 20 matching draws of the published equalizer (homology E1 draws, original data)
    er = one(e1, task=C, data="original")
    med = rc.p7_size_effect / (rc.A0 - er.kmer_auroc_median)
    hi = rc.p7_size_effect / (rc.A0 - er.kmer_auroc_max)
    lo = rc.p7_size_effect / (rc.A0 - er.kmer_auroc_min)
    macro("smDraws", intc(er.n_draws), "homology_recluster_e1_draws_summary.csv cohn original n_draws")
    macro("smDrawRange", f"{f(er.kmer_auroc_min)}--{f(er.kmer_auroc_max)}", "same, kmer_auroc_min/max")
    macro("smDrawMedian", f(er.kmer_auroc_median), "same, kmer_auroc_median")
    macro("smSizeShareMedian", pct(med), "derived: decision.csv cohn p7_size_effect / (A0 - e1 kmer_auroc_median)")
    macro("smSizeShareMedianExact", pct(med, 1), "same, one decimal")
    macro("smSizeShareMax", pct(hi), "derived: size_effect / (A0 - e1 kmer_auroc_max)")
    macro("smSizeShareMin", pct(lo), "derived: size_effect / (A0 - e1 kmer_auroc_min)")
    macro("smDrawMCC", f"{f(er.kmer_mcc_min)}--{f(er.kmer_mcc_max)}", "homology_recluster_e1_draws_summary.csv cohn original kmer_mcc_min/max")
    macro("smDrawMCCmed", f(er.kmer_mcc_median), "same, kmer_mcc_median")
    macro("smTau", f(rc.tau, 3), "decision.csv cohn tau")
    rd = dec.loc[D]
    macro("smCtlAzero", f(rd.A0), "decision.csv drosophila A0")
    macro("smCtlAr", f(rd.Ar_mean), "decision.csv drosophila Ar_mean")
    macro("smCtlAm", f(rd.Am), "decision.csv drosophila Am")
    macro("smCtlAmCI", ci(rd.Am_lo, rd.Am_hi), "same, CI")
    macro("smCtlComp", f(rd.comp0), "decision.csv drosophila comp0 (LightGBM composition-only)")
    macro("smCtlPubK", f(rd.paper_anchor_auroc), "decision.csv drosophila paper_anchor_auroc")
    macro("smCtlPubC", f(rd.comp_paper_anchor_auroc), "decision.csv drosophila comp_paper_anchor_auroc")
    macro("smCompZero", f(rc.comp0), "decision.csv cohn comp0")
    macro("smCompPub", f(rc.comp_paper_anchor_auroc), "decision.csv cohn comp_paper_anchor_auroc")
    macro("smCompPubCI", ci(rc.comp_paper_anchor_auroc_lo, rc.comp_paper_anchor_auroc_hi), "same, CI")

    # ---- full strictness sweep (F1, retrained within design)
    def srow(ds, lev, model):
        return one(summ, dataset=ds, family="F1_hist", level_label=lev, design="within", model=model)
    rows = []
    for ds in [C, NL, D]:
        rows.append(("group", DSL[ds]))
        levs = summ[(summ.dataset == ds) & (summ.family == "F1_hist") & (summ.design == "within") & (summ.model == "kmer")].sort_values("level")
        for _, r in levs.iterrows():
            c = srow(ds, r.level_label, "comp")
            rows.append([f(r.level, 2), pct(r.retained_neg_frac, 1), intc(r.n_test_neg), f(r.max_abs_smd_matched, 2),
                         f"{f(r.auroc_matched)} {ci(r.auroc_matched_lo, r.auroc_matched_hi)}", f"{f(r.auroc_matched_seedmin)}--{f(r.auroc_matched_seedmax)}",
                         f"{f(r.auroc_random_mean)} ({f(r.auroc_random_min)}--{f(r.auroc_random_max)})",
                         f(r.auroc_gap_random_minus_matched, 3, True), f(c.auroc_gap_random_minus_matched, 3, True)])
    write_table("tab_psm_sweep.tex", "tab:psmsweep",
                "Matching-strictness sweep: composition-matched vs.\\ size-matched random negatives at every level (retrained, "
                "within-arm design; $k$-mer AUROC).",
                ["psm_sensitivity_summary.csv"],
                "rrrrlllrr", ["$\\alpha$", "Neg.\\ kept", "Test neg.", "max$|$SMD$|$", "Matched, seed 42 [CI]", "Seeds 42--44",
                              "Random (range)", "Gap", "Comp.-only gap"], rows, size="\\scriptsize", tabcolsep="3.5pt",
                notes="$\\alpha$: share of the propensity-histogram mismatch removed ($\\alpha{=}0$ keeps every negative; $\\alpha{=}1$ is "
                      "the released 20-bin equalizer rule, re-drawn). Gap = random $-$ matched (positive: matching lowers AUROC more than "
                      "removing as many random negatives); comp.-only gap: the same for the composition-only LightGBM. Gaps are computed "
                      "from unrounded values. Each arm is scored on its own reduced test set, so this design is unpaired; the paired designs "
                      "are in Table~\\ref{tab:psmdesigns}. The size-matched random controls fall along this sweep (the size component), "
                      "whereas along the caliper sweep (Table~\\ref{tab:psmcaliper}) random 1:1 subsets stay nearly flat.")
    a05 = srow(C, "a0.5", "kmer")
    a09 = srow(C, "a0.9", "kmer")
    a002 = srow(C, "a0.02", "kmer")
    a1 = srow(C, "a1", "kmer")
    macro("smSweepGapLoose", f(a002.auroc_gap_random_minus_matched, 3, True), "psm_sensitivity_summary.csv cohn F1 a0.02 kmer gap")
    macro("smSweepKeptLoose", pct(a002.retained_neg_frac), "same, retained_neg_frac")
    macro("smSweepGapHalf", f(a05.auroc_gap_random_minus_matched, 3, True), "cohn F1 a0.5 gap")
    macro("smSweepKeptHalf", pct(a05.retained_neg_frac), "cohn F1 a0.5 retained")
    macro("smSweepGapNine", f(a09.auroc_gap_random_minus_matched, 3, True), "cohn F1 a0.9 gap")
    macro("smSweepKeptNine", pct(a09.retained_neg_frac, 1), "cohn F1 a0.9 retained")
    macro("smSweepGapOne", f(a1.auroc_gap_random_minus_matched, 3, True), "cohn F1 a1 gap")
    macro("smSweepRandLoose", f(a002.auroc_random_mean), "cohn F1 a0.02 auroc_random_mean")
    macro("smSweepRandOne", f(a1.auroc_random_mean), "cohn F1 a1 auroc_random_mean")
    macro("smMCCrandMax", f(max(srow(C, 'a0.05', 'kmer').mcc_random_mean, srow(C, 'a0.1', 'kmer').mcc_random_mean), 3),
          "cohn F1 a0.05/a0.1 mcc_random_mean (max)")
    macro("smMCCrandMin", f(min(srow(C, 'a0.05', 'kmer').mcc_random_mean, srow(C, 'a0.1', 'kmer').mcc_random_mean), 3), "same (min)")
    macro("smMCCzero", f(rc.mcc0, 3), "decision.csv cohn mcc0")
    macro("smMCCrandKeptHi", pct(srow(C, 'a0.05', 'kmer').retained_neg_frac), "cohn F1 a0.05 retained")
    macro("smMCCrandKeptRange", prange(srow(C, 'a0.1', 'kmer').retained_neg_frac, srow(C, 'a0.05', 'kmer').retained_neg_frac), "cohn F1 a0.1/a0.05 retained (range)")
    macro("smNrandom", intc(a1.n_random), "psm_sensitivity_summary.csv cohn a1 n_random")
    macro("smMCCrandKeptLo", pct(srow(C, 'a0.1', 'kmer').retained_neg_frac), "cohn F1 a0.1 retained")
    _rm = [srow(C, 'a0.05', 'kmer').mcc_random_mean, srow(C, 'a0.1', 'kmer').mcc_random_mean]
    macro("smMCCrandDrop", f(rc.mcc0 - float(np.mean(_rm)), 2), "derived: decision.csv cohn mcc0 - mean of F1 a0.05/a0.1 mcc_random_mean")

    # ---- caliper sweep
    rows = []
    for ds in [C, NL, D]:
        rows.append(("group", DSL[ds]))
        for _, r in cal[cal.dataset == ds].sort_values("caliper_sd", ascending=False).iterrows():
            rows.append([f(r.caliper_sd, 2).rstrip("0").rstrip("."), pct(r.retained_frac_each_class), f"{intc(r.n_test_pos)}/{intc(r.n_test_neg)}",
                         f(r.max_abs_smd_matched, 2), f(r.max_abs_smd_random_mean, 2),
                         f"{f(r.kmer_matched)} {ci(r.kmer_matched_lo, r.kmer_matched_hi)}", f"{f(r.kmer_random_mean)} ({f(r.kmer_random_min)}--{f(r.kmer_random_max)})",
                         f(r.kmer_gap, 3, True), f(r.comp_gap, 3, True)])
    write_table("tab_psm_caliper.tex", "tab:psmcaliper",
                "Caliper sweep: two-sided 1:1 nearest-neighbour matching on the logit propensity vs.\\ random 1:1 subsets of the "
                "same size ($k$-mer AUROC; pre-specified part 4b).",
                ["psm_sensitivity_caliper_sweep.csv"],
                "rrrrrllrr", ["Caliper (SD)", "Kept/class", "Test pos./neg.", "SMD matched", "SMD random", "Matched [CI]", "Random 1:1 (range)",
                              "Gap", "Comp.-only gap"], rows, size="\\scriptsize", tabcolsep="3.5pt",
                notes="Both classes are subsampled to the same fraction, so the test stays balanced. Kept/class: retained fraction of "
                      "negatives (for the leak-free \\task{nt\\_enhancers} test the positive fraction differs by at most about one "
                      "point, because the leak-free removal touches positives only). SMD: max$|$SMD$|$ over the 20 composition features.")
    c02 = one(cal, dataset=C, caliper_sd=0.2)
    c2 = one(cal, dataset=C, caliper_sd=2.0)
    n02 = one(cal, dataset=NL, caliper_sd=0.2)
    d02 = one(cal, dataset=D, caliper_sd=0.2)
    macro("smCalGapTwo", f(c2.kmer_gap, 3, True), "psm_sensitivity_caliper_sweep.csv cohn c2 kmer_gap")
    macro("smCalGapPtTwo", f(c02.kmer_gap, 3, True), "cohn c0.2 kmer_gap")
    macro("smCalMatchedPtTwo", f(c02.kmer_matched), "cohn c0.2 kmer_matched")
    macro("smCalRandPtTwo", f(c02.kmer_random_mean), "cohn c0.2 kmer_random_mean")
    cr = cal[cal.dataset == C]
    macro("smCalRandRange", f"{f(cr.kmer_random_mean.min())}--{f(cr.kmer_random_mean.max())}", "cohn kmer_random_mean over calipers")
    macro("smCalKeptPtTwo", pct(c02.retained_frac_each_class), "cohn c0.2 retained_frac_each_class")
    macro("smCalNtMatched", f"{f(n02.kmer_matched)} {ci(n02.kmer_matched_lo, n02.kmer_matched_hi)}", "nt leak-free c0.2 kmer_matched")
    macro("smCalNtRand", f(n02.kmer_random_mean), "nt leak-free c0.2 kmer_random_mean")
    macro("smCalNtGap", f(n02.kmer_gap, 3, True), "nt leak-free c0.2 kmer_gap")
    macro("smCalNtComp", f(n02.comp_gap, 3, True), "nt leak-free c0.2 comp_gap")
    macro("smCalNtKept", pct(n02.retained_frac_each_class), "nt leak-free c0.2 retained")
    macro("smCalNtSMD", f(n02.max_abs_smd_matched, 2), "nt leak-free c0.2 max_abs_smd_matched")
    macro("smCalCtlGap", f(d02.kmer_gap, 3, True), "control c0.2 kmer_gap")

    # ---- designs at the published retention: which are paired
    def drow(label, kind, cells):
        return [label, kind] + cells
    rows = []
    for ds in [C, NL, D]:
        r = dec.loc[ds]
        rows.append(("group", DSL[ds]))
        rows.append(drow("Retrained on matched vs.\\ random negatives, each scored on its own reduced test", "unpaired (within)",
                         [f(r.Am), f(r.Ar_mean), f(r.Ar_mean - r.Am, 3, True)]))
        rows.append(drow("Retrained on matched vs.\\ random negatives, scored on the untouched original test", "paired (undistorted test)",
                         [f(r.trainonly_A_matchedTrain), f(r.trainonly_A_randomTrain_mean), f(r.trainonly_A_randomTrain_mean - r.trainonly_A_matchedTrain, 3, True)]))
        rows.append(drow("Original model scored on the matched vs.\\ random test subsets", "paired (model fixed)",
                         [f(r.fixed_A_matchedTest), f(r.fixed_A_randomTest_mean), f(r.fixed_A_randomTest_mean - r.fixed_A_matchedTest, 3, True)]))
        rows.append(drow("Original model on the original test, class ratio changed only", "size/ratio check",
                         ["--", "--", f(r.test_distortion_effect, 3, True)]))
        ow_ok = bool(r.fixedoverlap_share_interpretable) if not isnan(r.fixedoverlap_share_interpretable) else False
        rows.append(drow("Original model, overlap-weighted original test vs.\\ permuted-weight placebo", "removal-free, paired",
                         [f(r.fixedoverlap_A) + ("" if ow_ok else "$^\\ddagger$"), f(r.fixedoverlap_placebo), f(r.fixedoverlap_placebo - r.fixedoverlap_A, 3, True)]))
        rows.append(drow("Retrained with overlap weights vs.\\ placebo", "removal-free",
                         [f(r.Aw_overlap) + ("" if not bool(r.overlap_weights_overcorrect) else "$^\\ddagger$"), f(r.Apw_placebo_mean), f(r.Apw_placebo_mean - r.Aw_overlap, 3, True)]))
        rows.append(drow("Positives matched to the negatives, every negative kept, vs.\\ random positive subsets", "no negative removed" + ("$^\\S$" if bool(r.f4_underpowered) else ""),
                         [f(r.f4_Am), f(r.f4_Ar_mean), f(r.f4_Ar_mean - r.f4_Am, 3, True)]))
        rows.append(drow("Two-sided 1:1 caliper 0.2 SD vs.\\ random 1:1", "balanced test",
                         [f(r.A2_1to1), f(r.A2r_mean), f(r.A2r_mean - r.A2_1to1, 3, True)]))
    write_table("tab_psm_designs.tex", "tab:psmdesigns",
                "Designs at $\\alpha{=}1$ that change what matching can distort: which test set is scored, whether the "
                "model is retrained, and whether any negative is removed ($k$-mer AUROC; composition-balanced: matched with selection "
                "seed 42, or overlap-weighted; random / placebo: mean of 10 size-matched random draws, or of the permuted-weight "
                "placebo for the weighting rows).",
                ["psm_sensitivity_decision.csv"],
                ">{\\raggedright\\arraybackslash}p{0.46\\linewidth}>{\\raggedright\\arraybackslash}p{0.17\\linewidth}rrr",
                ["Design", "Pairing", "Composition-balanced", "Random / placebo", "Difference"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes="`Paired' designs hold either the model or the test set fixed, so a difference cannot come from a changed test "
                      "distribution (undistorted test) or from retraining on fewer negatives (model fixed). $^\\ddagger$ the overlap "
                      "weights over-correct (the composition-only model's weighted AUROC falls below 0.5), so only the direction is "
                      "interpretable. $^\\S$ underpowered ("
                      f"{intc(dec.loc[NL].f4_n_test_pos)} test positives). Differences are computed from unrounded values; CIs of the "
                      "within, undistorted-test and model-fixed contrasts are in Table~\\ref{tab:psmdecomp}.")
    macro("smTrainOnly", f(rc.trainonly_A_matchedTrain), "decision.csv cohn trainonly_A_matchedTrain")
    macro("smTrainOnlyR", f(rc.trainonly_A_randomTrain_mean), "decision.csv cohn trainonly_A_randomTrain_mean")
    macro("smTrainOnlyD", f"{f(rc.p7_trainonly_matched_vs_random, 3, True)} {ci(rc.p7_trainonly_matched_vs_random_lo, rc.p7_trainonly_matched_vs_random_hi, 3, True)}", "decision.csv cohn p7_trainonly_matched_vs_random")
    macro("smFixed", f(rc.fixed_A_matchedTest), "decision.csv cohn fixed_A_matchedTest")
    macro("smFixedR", f(rc.fixed_A_randomTest_mean), "decision.csv cohn fixed_A_randomTest_mean")
    macro("smFixedD", f"{f(rc.p7_fixed_matched_vs_random, 3, True)} {ci(rc.p7_fixed_matched_vs_random_lo, rc.p7_fixed_matched_vs_random_hi, 3, True)}", "decision.csv cohn p7_fixed_matched_vs_random")
    macro("smRatioOnly", f(rc.test_distortion_effect, 3, True), "decision.csv cohn test_distortion_effect")
    macro("smFfour", f(rc.f4_Am), "decision.csv cohn f4_Am")
    macro("smFfourCI", ci(rc.f4_Am_lo, rc.f4_Am_hi), "same, CI")
    macro("smFfourR", f(rc.f4_Ar_mean), "decision.csv cohn f4_Ar_mean")
    macro("smFfourPos", pct(rc.f4_retained_pos_frac, 1), "decision.csv cohn f4_retained_pos_frac")
    macro("smOverlapFixed", f(rc.fixedoverlap_A), "decision.csv cohn fixedoverlap_A")
    macro("smOverlapPlacebo", f(rc.fixedoverlap_placebo), "decision.csv cohn fixedoverlap_placebo")
    macro("smOneToOne", f(rc.A2_1to1), "decision.csv cohn A2_1to1")
    macro("smOneToOneR", f(rc.A2r_mean), "decision.csv cohn A2r_mean")
    rn = dec.loc[NL]
    macro("smNtFixedD", f"{f(rn.p7_fixed_matched_vs_random, 3, True)} {ci(rn.p7_fixed_matched_vs_random_lo, rn.p7_fixed_matched_vs_random_hi, 3, True)}", "decision.csv nt leak-free p7_fixed_matched_vs_random")
    macro("smNtTrainD", f"{f(rn.p7_trainonly_matched_vs_random, 3, True)} {ci(rn.p7_trainonly_matched_vs_random_lo, rn.p7_trainonly_matched_vs_random_hi, 3, True)}", "decision.csv nt leak-free p7_trainonly_matched_vs_random")
    macro("smNtWithinD", f"{f(rn.p7_matched_vs_random, 3, True)} {ci(rn.p7_matched_vs_random_lo, rn.p7_matched_vs_random_hi, 3, True)}", "decision.csv nt leak-free p7_matched_vs_random")
    macro("smNtFixedComp", f(rn.p7_fixed_comp_matched_vs_random, 3, True), "decision.csv nt leak-free p7_fixed_comp_matched_vs_random")
    macro("smNtTrainComp", f(rn.p7_trainonly_comp_matched_vs_random, 3, True), "decision.csv nt leak-free p7_trainonly_comp_matched_vs_random")
    macro("smNtAm", f"{f(rn.Am)} {ci(rn.Am_lo, rn.Am_hi)}", "decision.csv nt leak-free Am")
    macro("smNtAr", f(rn.Ar_mean), "decision.csv nt leak-free Ar_mean")
    macro("smNtSMD", f(rn.f1_max_abs_smd, 2), "decision.csv nt leak-free f1_max_abs_smd")
    macro("smNtSMDredraw", f(dec.loc[N].f1_max_abs_smd, 2), "decision.csv nt uncorrected f1_max_abs_smd")
    # the pre-specified nt_enhancers test is the within design on the ORIGINAL (uncorrected) test
    rp = dec.loc[N]
    macro("smNtPreAm", f"{f(rp.Am)} {ci(rp.Am_lo, rp.Am_hi)}", "decision.csv nt_enhancers (original test, pre-specified) Am")
    macro("smNtPreAr", f(rp.Ar_mean), "decision.csv nt_enhancers (original test) Ar_mean")
    macro("smNtPreWithinD", f"{f(rp.p7_matched_vs_random, 3, True)} {ci(rp.p7_matched_vs_random_lo, rp.p7_matched_vs_random_hi, 3, True)}",
          "decision.csv nt_enhancers (original test) p7_matched_vs_random")

    # ---- P7 decomposition with CIs
    rows = []
    dlab = {"within": "within (retrained, own test)", "train_only": "undistorted test (retrained, original test)", "fixed_model": "model fixed (original model)"}
    mlab = {"kmer": "$k$-mer", "comp": "composition-only", "kmer_minus_comp": "$k$-mer $-$ comp.-only"}
    for ds in [C, NL, D]:
        rows.append(("group", DSL[ds]))
        for des in ["within", "train_only", "fixed_model"]:
            for seeds in (["42", "42-44"] if des == "within" else ["42"]):
                for model in ["kmer", "comp"]:
                    q = dci[(dci.dataset == ds) & (dci.design == des) & (dci.model == model) & (dci.matched_seeds.astype(str) == seeds)]
                    if not len(q):
                        continue
                    r = q.iloc[0]
                    size_cell = "--" if des == "fixed_model" else f"{f(r.size_effect, 3, True)} {ci(r.size_effect_lo, r.size_effect_hi, 3, True)}"
                    rows.append([dlab[des] + ("" if seeds == "42" else ", seeds 42--44"), mlab[model], size_cell,
                                 f"{f(r.matched_vs_random, 3, True)} {ci(r.matched_vs_random_lo, r.matched_vs_random_hi, 3, True)}",
                                 intc(r.n_test_neg_scored)])
    write_table("tab_psm_decomp.tex", "tab:psmdecomp",
                "Decomposition of the matching drop at $\\alpha{=}1$, with two-level paired bootstrap CIs (post hoc P7): "
                "size effect (original $-$ random) and matched-vs-random effect (random $-$ matched).",
                ["psm_sensitivity_decomposition_ci.csv"],
                "llllr", ["Design", "Model", "Size effect [CI]", "Matched-vs-random [CI]", "Test neg.\\ scored"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Bootstrap: test sequences (shared by every arm) $\\times$ random seeds, 1000 resamples; the models are fixed, so "
                      "training-set variability is not resampled. In the model-fixed design the original model is scored on the matched "
                      "vs.\\ random test subsets, so there is no size term. The composition-only rows show how much composition signal the "
                      "same selection removes; they are the reference for reading the $k$-mer rows. Within-design rows use one matching "
                      "draw (seed 42) unless marked.")
    gd = lambda ds, des, model, seeds="42": dci[(dci.dataset == ds) & (dci.design == des) & (dci.model == model) & (dci.matched_seeds.astype(str) == seeds)].iloc[0]
    for des, nm in [("within", "Within"), ("fixed_model", "Fixed"), ("train_only", "Train")]:
        k = gd(D, des, "kmer"); cmp = gd(D, des, "comp")
        macro(f"smCtl{nm}K", f(k.matched_vs_random, 3), f"decomposition_ci drosophila {des} kmer seed 42 matched_vs_random")
        macro(f"smCtl{nm}C", f(cmp.matched_vs_random, 3), f"decomposition_ci drosophila {des} comp seed 42 matched_vs_random")
    kc = gd(D, "within", "comp")
    macro("smCtlWithinCCI", ci(kc.matched_vs_random_lo, kc.matched_vs_random_hi, 3, True), "decomposition_ci drosophila within comp seed 42 CI")
    macro("smCtlWithinKS", f(gd(D, "within", "kmer", "42-44").matched_vs_random, 3), "decomposition_ci drosophila within kmer 42-44")
    macro("smCtlWithinCS", f(gd(D, "within", "comp", "42-44").matched_vs_random, 3), "decomposition_ci drosophila within comp 42-44")
    kk = gd(C, "within", "comp")
    macro("smCohnWithinC", f(kk.matched_vs_random, 3), "decomposition_ci cohn within comp seed 42")
    macro("smCohnFixedK", f(gd(C, "fixed_model", "kmer").matched_vs_random, 3), "decomposition_ci cohn fixed kmer")
    macro("smCohnFixedC", f(gd(C, "fixed_model", "comp").matched_vs_random, 3), "decomposition_ci cohn fixed comp")

    # ---- selection bias and label noise
    sb = sel[(sel.dataset == C) & (sel.selection == "F1_hist_a1_seed42")]
    feats = [("tata_any", "TATA-box hit (TBP $\\ge$1.3 bits/pos)"), ("ccre_els", "ENCODE enhancer-like cCRE (ELS) overlap"),
             ("ccre_any", "any ENCODE cCRE overlap"), ("ccre_pls", "promoter-like cCRE (PLS) overlap"),
             ("tss500_overlap", "GENCODE v44 TSS $\\pm$500\\,bp overlap"), ("repeatmasker_frac", "RepeatMasker coverage")]
    rows = []
    for fk, fl in feats:
        q = sb[sb.feature == fk]
        if not len(q):
            continue
        r = q.iloc[0]
        rows.append([fl, f(r.mean_retained, 3), f(r.mean_discarded, 3), f(r.mean_random_sizematched, 3), f(r.mean_pos, 3), sig2(r.p_bonferroni)])
    rtata = one(sb, feature="tata_any")
    write_table("tab_psm_selbias.tex", "tab:psmselbias",
                "Selection bias of composition matching on \\task{cohn} ($\\alpha{=}1$, seed 42): features outside the "
                "propensity model in kept vs.\\ discarded negatives.",
                ["psm_sensitivity_selection_bias.csv"], "lrrrrr",
                ["Feature", f"Kept neg.\\ ($n{{=}}{intc(rtata.n_retained)}$)", f"Discarded ({intc(rtata.n_discarded)})", "Random, same size", "Positives", "Bonferroni $p$"],
                rows, notes=f"Fisher or Mann--Whitney test of kept vs.\\ discarded negatives, Bonferroni-adjusted over {intc(rtata.n_tests_bonferroni)} tests "
                            "(part 6, pre-specified). The kept negatives match the positives' TATA rate and are enriched for enhancer-like "
                            "cCREs, the expected signature if some are unlabeled enhancers; Table~\\ref{tab:psmlabelnoise} checks what that "
                            "could do to the AUROC.")
    rels = one(sb, feature="ccre_els")
    rany = one(sb, feature="ccre_any")
    rtss = one(sb, feature="tss500_overlap")
    macro("smElsKept", pct(rels.mean_retained), "psm_sensitivity_selection_bias.csv cohn F1_hist_a1_seed42 ccre_els mean_retained")
    macro("smElsDisc", pct(rels.mean_discarded), "same, mean_discarded")
    macro("smElsPos", pct(rels.mean_pos), "same, mean_pos")
    macro("smElsP", sig2(rels.p_bonferroni), "same, p_bonferroni")
    macro("smAnyKept", pct(rany.mean_retained), "same file, ccre_any mean_retained")
    macro("smAnyDisc", pct(rany.mean_discarded), "same, ccre_any mean_discarded")
    macro("smTataKept", f(rtata.mean_retained, 3), "same file, tata_any mean_retained")
    macro("smTataDisc", f(rtata.mean_discarded, 3), "same, mean_discarded")
    macro("smTataPos", f(rtata.mean_pos, 3), "same, mean_pos")
    macro("smTssKept", f(rtss.mean_retained, 3), "same file, tss500_overlap mean_retained")
    macro("smTssDisc", f(rtss.mean_discarded, 3), "same, mean_discarded")
    macro("smTssP", sig2(rtss.p_bonferroni), "same, p_bonferroni")

    rows = [["cCRE-overlapping test negatives removed, same models (pre-specified design c)", f"{f(rc.ccrefreeTest_Am)} {ci(rc.ccrefreeTest_Am_lo, rc.ccrefreeTest_Am_hi)}",
             f(rc.ccrefreeTest_Ar_mean), intc(rc.ccrefreeTest_n_neg_matched)],
            ["Two-sided 1:1 on the cCRE-purged pool (design b; balanced)", f"{f(rc.f5b_Am)} {ci(rc.f5b_Am_lo, rc.f5b_Am_hi)}", f(rc.f5b_Ar_mean), "--"],
            ["Re-matching on the cCRE-purged pool (design a; not balanced)", f"{f(rc.f5_Am)} {ci(rc.f5_Am_lo, rc.f5_Am_hi)}", f(rc.f5_Ar_mean), "--"]]
    write_table("tab_psm_labelnoise.tex", "tab:psmlabelnoise",
                "Label-noise checks on \\task{cohn}: does matching lower AUROC only because kept negatives are unlabeled enhancers?",
                ["psm_sensitivity_decision.csv"], ">{\\raggedright\\arraybackslash}p{0.5\\linewidth}llr",
                ["Check", "Matched [CI]", "Random", "Matched test neg."], rows,
                notes=f"Design (a) keeps {pct(rc.f5_retained_neg_frac, 1)} of negatives and is not composition-balanced (max$|$SMD$|$ "
                      f"{f(rc.f5_max_abs_smd, 2)}); design (b) is balanced (max$|$SMD$|$ {f(rc.f5b_max_abs_smd, 2)}). A mixture-model "
                      f"\\emph{{estimate}} (not a worst case) of the AUROC cost of hidden enhancers among the kept negatives is "
                      f"{f(rc.labelnoise_max_auroc_drop, 3)}, {f(rc.labelnoise_bound_over_shortcut_drop, 2)}$\\times$ the matched-vs-random effect.")
    macro("smCcreFree", f(rc.ccrefreeTest_Am), "decision.csv cohn ccrefreeTest_Am")
    macro("smCcreFreeCI", ci(rc.ccrefreeTest_Am_lo, rc.ccrefreeTest_Am_hi), "same, CI")
    macro("smCcreFreeR", f(rc.ccrefreeTest_Ar_mean), "decision.csv cohn ccrefreeTest_Ar_mean")
    macro("smCcreFreeN", intc(rc.ccrefreeTest_n_neg_matched), "decision.csv cohn ccrefreeTest_n_neg_matched")
    macro("smLabelNoise", f(rc.labelnoise_max_auroc_drop, 3), "decision.csv cohn labelnoise_max_auroc_drop")
    macro("smLabelNoiseRatio", f(rc.labelnoise_bound_over_shortcut_drop, 2), "decision.csv cohn labelnoise_bound_over_shortcut_drop")

    # ---- MCC readout
    rows = []
    for ds in [C, NL, D]:
        r = dec.loc[ds]
        rows.append([DSL[ds], f(r.mcc0), f"{f(r.mcc_r_mean)} ({f(r.mcc_r_min)}--{f(r.mcc_r_max)})", f(r.paper_anchor_mcc), f(r.mcc_collapse_share_reproduced_by_random, 2),
                     f(r.comp_mcc0), f(r.comp_mcc_r_mean), f(r.comp_paper_anchor_mcc), f(r.comp_mcc_collapse_share_reproduced_by_random, 2)])
    write_table("tab_psm_mcc.tex", "tab:psmmcc",
                "Why the equalized-split MCC is not used: size-matched random negatives reproduce most or all of its collapse "
                "(MCC at threshold 0.5).",
                ["psm_sensitivity_decision.csv"], ">{\\raggedright\\arraybackslash}p{0.2\\linewidth}rlrrrrrr",
                [["", "\\multicolumn{4}{c}{$k$-mer LightGBM}", "\\multicolumn{4}{c}{composition-only LightGBM}"],
                 ["Data", "Original", "Random (range)", "Released split", "Share", "Original", "Random", "Released split", "Share"]],
                rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Share = (original $-$ random)/(original $-$ released equalized split), the share of the collapse that size-matched random negatives reproduce. "
                      f"On the \\task{{cohn}} test set, which is about {pct(a1.n_test_pos / (a1.n_test_pos + a1.n_test_neg))} positive at this retention, a model "
                      f"trained on about {pct(a1.n_train_pos / (a1.n_train_pos + a1.n_train_neg))} positives predicts almost everything positive, so MCC at 0.5 mostly measures the size/imbalance "
                      f"change. Over the {intc(er.n_draws)} equally valid matching draws of the released equalizer on \\task{{cohn}}, the equalized $k$-mer MCC "
                      f"ranges {f(er.kmer_mcc_min)}--{f(er.kmer_mcc_max)} (median {f(er.kmer_mcc_median)}; "
                      "\\task{homology\\_recluster\\_e1\\_draws\\_summary.csv}). In the balanced 1:1 design MCC isolates the "
                      f"matched-vs-random contrast at caliper 0.2 SD: {f(c02.mcc05_matched)} vs.\\ {f(c02.mcc05_random_mean)} for random 1:1 subsets "
                      "(\\task{psm\\_sensitivity\\_caliper\\_sweep.csv}).")
    macro("smMCCrand", f(rc.mcc_r_mean), "decision.csv cohn mcc_r_mean")
    macro("smMCCrandRange", f"{f(rc.mcc_r_min)}--{f(rc.mcc_r_max)}", "same, min/max")
    macro("smMCCpub", f(rc.paper_anchor_mcc), "decision.csv cohn paper_anchor_mcc")
    macro("smMCCshare", f(rc.mcc_collapse_share_reproduced_by_random, 2), "decision.csv cohn mcc_collapse_share_reproduced_by_random")
    macro("smCompMCCzero", f(rc.comp_mcc0), "decision.csv cohn comp_mcc0")
    macro("smCompMCCrand", f(rc.comp_mcc_r_mean), "decision.csv cohn comp_mcc_r_mean")
    macro("smCompMCCshare", pct(rc.comp_mcc_collapse_share_reproduced_by_random), "decision.csv cohn comp_mcc_collapse_share_reproduced_by_random")
    macro("smCompMCCpub", f(rc.comp_paper_anchor_mcc), "decision.csv cohn comp_paper_anchor_mcc")

    # ---- between-dataset contrast
    bd = bet[bet.kind == "dataset"]
    bcn = bet[bet.kind == "contrast"]
    rows = []
    for des, dl in [("fixed_model_overlap", "original model, overlap-weighted test"), ("within_overlap", "retrained with overlap weights")]:
        rows.append(("group", dl))
        for ds in [C, D]:
            r = one(bd, dataset=ds, design=des)
            rows.append([DSL[ds], f"{f(r.share_above_chance_lost)} {ci(r.share_lo, r.share_hi)}", f"{f(r.comp_share_lost)} {ci(r.comp_share_lo, r.comp_share_hi)}",
                         f"{f(r.kmer_over_comp_share, 2)} {ci(r.kmer_over_comp_share_lo, r.kmer_over_comp_share_hi, 2)}"])
        r = one(bcn, contrast=f"{C} - {D}", design=des)
        rows.append(["\\task{cohn} $-$ control", f"{f(r.share_above_chance_lost, 3, True)} {ci(r.share_lo, r.share_hi, 3, True)}",
                     f"{f(r.comp_share_diff, 3, True)} {ci(r.comp_share_diff_lo, r.comp_share_diff_hi, 3, True)}",
                     f"diff.\\ of contrasts {f(r.diff_of_contrasts, 3, True)} {ci(r.diff_of_contrasts_lo, r.diff_of_contrasts_hi, 3, True)}"])
    write_table("tab_psm_between.tex", "tab:psmbetween",
                "Between-dataset contrast under removal-free overlap weighting: share of above-chance AUROC lost by the $k$-mer model "
                "and by the composition-only model under the same weights (post hoc P4/P6).",
                ["psm_sensitivity_between_dataset.csv"], "llll",
                ["Data", "$k$-mer share lost [CI]", "Comp.-only share lost [CI]", "$k$-mer/comp.-only [CI]"], rows, size="\\footnotesize",
                notes="The $k$-mer contrast between \\task{cohn} and the GC-balanced control is matched by the composition-only contrast "
                      "under the same weights (difference of contrasts near 0), so it measures reliance on the composition gap, not "
                      "contamination specificity. \\task{nt\\_enhancers} is omitted: the weights over-correct there (weighted AUROC below "
                      "0.5), so its shares ($>$1) are not interpretable.")
    r = one(bcn, contrast=f"{C} - {D}", design="fixed_model_overlap")
    macro("smBetK", f(r.share_above_chance_lost, 3, True), "between_dataset.csv contrast fixed_model_overlap share")
    macro("smBetC", f(r.comp_share_diff, 3, True), "same, comp_share_diff")
    macro("smBetDoc", f"{f(r.diff_of_contrasts, 3, True)} {ci(r.diff_of_contrasts_lo, r.diff_of_contrasts_hi, 3, True)}", "same, diff_of_contrasts")

    # ---- dissociation
    rows = []
    for ds in [C, NL, D]:
        for des in ["within", "fixed_model", "train_only"]:
            r = one(dis, dataset=ds, design=des)
            rows.append([DSL[ds] if des == "within" else "", dlab[des], intc(r.n_arms), f(r.spearman_kmer_gap_vs_comp_gap, 2, True),
                         f(r.spearman_kmer_gap_vs_neglog_retention, 2, True), f(r.ols_beta_comp_gap_per_sd, 3, True),
                         f(r.ols_beta_neglog_retention_per_sd, 3, True), esc(r.reading)])
    write_table("tab_psm_dissoc.tex", "tab:psmdissoc",
                "Across matching arms, does the $k$-mer gap track the composition signal removed or the selection depth? "
                "(post hoc, descriptive).",
                ["psm_sensitivity_dissociation_stats.csv"],
                "l>{\\raggedright\\arraybackslash}p{0.17\\linewidth}rrrrr>{\\raggedright\\arraybackslash}p{0.2\\linewidth}",
                ["Data", "Design", "Arms", "$\\rho$ w/ comp.\\ gap", "$\\rho$ w/ depth", "OLS comp.", "OLS depth", "Reading (0.3-margin rule)"],
                rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Spearman correlation of the $k$-mer gap with the composition-only gap and with selection depth ($-\\log$ retained "
                      "fraction) over the non-underpowered strictness, trimming and caliper arms; OLS: partial coefficients per SD. Arms "
                      "overlap and the random controls are nested, so no $p$-values are given. For \\task{cohn} the correlation with the "
                      "composition gap is the larger of the two in all three designs, but by the 0.3-margin rule the separation is clear only "
                      "in the within design.")

    # ---- figures
    copy_fig(RU / "figures" / "psm_sensitivity.pdf", "psm_sensitivity.pdf")
    copy_fig(RU / "figures" / "psm_sensitivity_caliper.pdf", "psm_sensitivity_caliper.pdf")
    copy_fig(RU / "figures" / "psm_sensitivity_ntraw.pdf", "psm_sensitivity_ntraw.pdf")

    # ---- control-motif panel (former Fig. 3) and threshold invariance
    sp = load("route3_specificity.csv")
    ti = load("route3_threshold_invariance.csv")
    sn0 = load("shuffled_neg.csv")
    assert int(sp.n_pos.max()) == int(ti.n_pos.max()) == int(sp.n_neg.max())
    hs = load("homology_recluster_extra_specificity.csv")
    rows = []
    npan = lambda c: int(sp[sp.motif_class == c].n_panel.iloc[0])
    cls = {"real_TATA": "real TATA (TBP PSSM)", "scrambled_TBP_panel": f"column-scrambled TBP ({npan('scrambled_TBP_panel')} PWMs)",
           "random_IC_panel": f"IC-matched random PWMs ({npan('random_IC_panel')})"}
    macro("smNscr", intc(npan("scrambled_TBP_panel")), "route3_specificity.csv scrambled_TBP_panel n_panel")
    macro("smNric", intc(npan("random_IC_panel")), "route3_specificity.csv random_IC_panel n_panel")
    for ds in ["human_enhancers_cohn", "nt_enhancers", "nt_promoter_tata"]:
        for _, r in sp[sp.dataset == ds].iterrows():
            val = f"{f(r.motif_only_auroc)} $\\pm$ {f(r.auroc_sd)}" if not isnan(r.auroc_sd) else f"{f(r.motif_only_auroc)} {ci(r.auroc_ci_lo, r.auroc_ci_hi)}"
            rows.append([short(ds), cls.get(r.motif_class, esc(r.motif_class)), val])
    tir = ti[(ti.split == "comp_equalized")]
    rows.append("midrule")
    for ds in ["human_enhancers_cohn", "nt_enhancers"]:
        q = tir[tir.dataset == ds]
        q0 = ti[(ti.split == "original") & (ti.dataset == ds)]
        rows.append([short(ds), "TATA AUROC over 0.8--1.5 bits/pos: original / equalized",
                     f"{f(q0.tata_auroc.min())}--{f(q0.tata_auroc.max())} / {f(q.tata_auroc.min())}--{f(q.tata_auroc.max())}"])
    write_table("tab_ctlmotif.tex", "tab:ctlmotif",
                "Control-motif panel (motif-only AUROC at 1.3 bits/pos; mean $\\pm$ SD over the panel, or estimate [95\\% CI]) and "
                "threshold invariance of the TATA skew.",
                ["route3_specificity.csv", "route3_threshold_invariance.csv"], "llr",
                ["Data", "Motif", "Motif-only AUROC"], rows,
                notes="An AUROC below 0.5 means the motif is commoner in the negatives. Random PWMs matched to the TBP information content "
                      "produce no class skew, so the scan does not manufacture one; column-scrambled TBP PWMs keep the TBP columns' AT-rich "
                      "composition but not the ordered TATA pattern and stay skewed, so the skew is composition-driven. Real TATA on "
                      "\\task{nt\\_promoter\\_tata} is skewed toward the positives (positive control). The panel was re-run on "
                      "homology-clean data (\\task{homology\\_recluster\\_extra\\_specificity.csv}); see Table~\\ref{tab:homroutes}. "
                      f"Both panels scan a seeded sample of at most {intc(sp.n_pos.max())} sequences per class (the equalized splits keep "
                      "fewer negatives); on all sequences the real-TATA AUROC of the original split is "
                      f"{f(one(sn0, dataset='human_enhancers_cohn', arm='original_neg').tata_auroc)} (\\task{{cohn}}) and "
                      f"{f(one(sn0, dataset='nt_enhancers', arm='original_neg').tata_auroc)} (\\task{{nt\\_enhancers}}; "
                      "Table~\\ref{tab:shuffle}).")
    macro("smRandICcohn", f"{f(one(sp, dataset='human_enhancers_cohn', motif_class='random_IC_panel').motif_only_auroc)}$\\pm${f(one(sp, dataset='human_enhancers_cohn', motif_class='random_IC_panel').auroc_sd)}", "route3_specificity.csv")
    # SM-local re-plot (the experiment's own panel functions on the same CSVs; only the panel-(b) legend is moved off the lines)
    plot_route3_panel("route3_control_motif_panel.pdf")
    stale = FIGS / "route3_control_motif_panel.png"
    if stale.exists():
        stale.unlink()

    # ---- shuffle hash-seed dependence
    hh = load("alt_model_shuffle_hashseed.csv")
    sn = load("shuffled_neg.csv")
    rows = []
    for ds in ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]:
        q = hh[hh.task == ds]
        s_or = one(sn, dataset=ds, arm="original_neg")
        s_sh = one(sn, dataset=ds, arm="shuffled_neg")
        rows.append([short(ds), f(s_or.tata_auroc), f"{f(s_sh.tata_auroc)} ({f(q.tata_auroc.min())}--{f(q.tata_auroc.max())})",
                     f"{f(s_sh.gc_auroc)}", f"{f(s_sh.kmer_auroc)} ({f(q.kmer_auroc.min())}--{f(q.kmer_auroc.max())})",
                     f"{f(s_sh.kmer_mcc)} ({f(q.kmer_mcc.min())}--{f(q.kmer_mcc.max())})", f"{int(q.reproduces_paper_csv.sum())}/{len(q)}"])
    write_table("tab_shuffle.tex", "tab:shuffle",
                f"Dinucleotide-shuffled negatives: the paper's draw and the range over {intc(hh.groupby('task').size().min())} \\task{{PYTHONHASHSEED}} draws of the shuffle.",
                ["shuffled_neg.csv", "alt_model_shuffle_hashseed.csv"], "lllllll",
                [["", "\\multicolumn{2}{c}{TATA motif-only AUROC}", "", "", "", ""],
                 ["Data", "real neg.", "shuffled (range)", "GC-content AUROC", "$k$-mer AUROC (range)", "$k$-mer MCC (range)", "= paper"]],
                rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Shuffles of the positives remove the mono- and dinucleotide composition gap by construction (GC-content AUROC "
                      f"{f(sn[sn.arm == 'shuffled_neg'].gc_auroc.min())} on every task; the GC-content AUROC is that of per-sequence GC content, "
                      "not a motif score). The negative-class TATA skew disappears (\\task{cohn} TATA AUROC slightly above 0.5 in every "
                      "draw), while the positives remain separable from their own shuffles by higher-order structure; for the control, whose "
                      "TATA is a positive-class motif, the TATA AUROC rises. All sequences are scanned. The shuffle depends on "
                      "\\task{PYTHONHASHSEED}; the paper's exact draw was not recovered by the "
                      f"{intc(hh.groupby('task').size().min())}-seed scan.")
    assert sn[sn.arm == "shuffled_neg"].gc_auroc.min() == sn[sn.arm == "shuffled_neg"].gc_auroc.max()
    s_c = one(sn, dataset="human_enhancers_cohn", arm="shuffled_neg")
    s_n = one(sn, dataset="nt_enhancers", arm="shuffled_neg")
    q = hh[hh.task == "human_enhancers_cohn"]
    macro("smShufTata", f(s_c.tata_auroc), "shuffled_neg.csv cohn shuffled_neg tata_auroc")
    macro("smShufTataRange", f"{f(q.tata_auroc.min())}--{f(q.tata_auroc.max())}", "alt_model_shuffle_hashseed.csv cohn tata_auroc range")
    macro("smShufKmerCohn", f(s_c.kmer_auroc, 2), "shuffled_neg.csv cohn shuffled_neg kmer_auroc")
    macro("smShufKmerNt", f(s_n.kmer_auroc, 2), "shuffled_neg.csv nt shuffled_neg kmer_auroc")
    macro("smHashDraws", intc(hh.groupby("task").size().min()), "alt_model_shuffle_hashseed.csv draws per task")

    # leakage
    lr_ = one(lk, dataset="nt_enhancers")
    macro("smNtDup", intc(lr_.n_test_verbatim_in_train), "psm_sensitivity_leakage_check.csv nt n_test_verbatim_in_train")
    macro("smNtTest", intc(lr_.n_test), "same, n_test")
    macro("smNtLeakA", f(lr_.orig_auroc_full_test), "same, orig_auroc_full_test")
    macro("smNtLeakFreeA", f(lr_.orig_auroc_leakfree_test), "same, orig_auroc_leakfree_test")


# ============================================================================ S5
def s5():
    hi = load("fm_head_invariance.csv")
    ca = load("fm_cross_architecture.csv")
    def pci_str(sci):
        """'[-0.003,0.014]' (the file's CI string) -> signed CI with the SM's minus sign and explicit plus."""
        lo, hi = (float(x) for x in str(sci).strip("[] ").split(","))
        return ci(lo, hi, 3, True)

    def head_name(h):
        assert h in ("lgbm", "lr"), h
        return "LightGBM" if h == "lgbm" else "LR"

    rows = []
    # NB: r["head"], not r.head (the latter is the bound pandas method Series.head)
    for _, r in hi.iterrows():
        rows.append(["HyenaDNA-tiny", short(r.dataset), "[SEP]" if r.pooling == "sep" else "mean", head_name(r["head"]),
                     f"{f(r.test_effect_auroc, 4, True)} {pci_str(r.test_effect_auroc_ci)}", "yes" if bool(r.significant) else "no"])
    for _, r in ca[ca.architecture.str.startswith("DNABERT")].iterrows():
        rows.append(["DNABERT-2-117M", short(r.dataset), "[CLS]" if r.pooling == "cls" else "mean", head_name(r["head"]),
                     f"{f(r.test_effect_auroc, 4, True)} {pci_str(r.ci)}", "yes" if bool(r.significant) else "no"])
    assert sum(x[3] == "LightGBM" for x in rows) == sum(x[3] == "LR" for x in rows) == len(rows) // 2
    write_table("tab_frozen.tex", "tab:frozen",
                "Frozen FM probes: test effect, the AUROC change of the same fixed head (negative = loss) when composition-biased "
                "test negatives are removed (GC-matched arm), by pooling and head.",
                ["fm_head_invariance.csv", "fm_cross_architecture.csv"], "llllll",
                ["Model", "Data", "Pooling", "Head", "Test effect [95\\% CI]", "CI excludes 0"], rows, size="\\footnotesize",
                notes="Nested paired bootstrap. The effect has the same sign and significance for both heads under each pooling "
                      "(head-invariant). Whether the GC-balanced control (\\task{drosophila\\_enhancers\\_stark}) is flat depends on the "
                      "pooling: it is flat under the [SEP] token only. DNABERT-2 was not run on the control; its flat reference in the paper "
                      "is a placebo that removes as many \\task{cohn} test negatives at random.")
    rr = one(hi, dataset="drosophila_enhancers_stark", pooling="sep", head="lgbm")
    rm = one(hi, dataset="drosophila_enhancers_stark", pooling="mean", head="lgbm")
    macro("smFrozCtlSep", f(rr.test_effect_auroc, 3, True), "fm_head_invariance.csv drosophila sep lgbm")
    gpl = load("fm_finetune_v2_gc_paired.csv")
    gm = one(gpl[gpl.kind == "level"], dataset="drosophila_enhancers_stark", model="hyenadna_frozen_mean")
    macro("smFrozCtlMean", f(gm.G_minus_A, 3, True), "fm_finetune_v2_gc_paired.csv level drosophila hyenadna_frozen_mean G_minus_A (unrounded -0.0385)")
    # SM-local vector re-plot from fm_cross_architecture.csv (8-pt text; replaces the cropped 300-dpi PNG copy)
    plot_frozen_probe("fm_frozen_probe.pdf")
    stale = FIGS / "fm_frozen_probe.png"
    if stale.exists():
        stale.unlink()

    # ---- fine-tuning arms
    pt = load("fm_finetune_v2_paper_table.csv")
    ud = load("fm_finetune_v2_unit_diagnostics.csv")
    rows = []
    mname = {"hyenadna": "HyenaDNA-tiny, full", "dnabert2": "DNABERT-2, LoRA"}
    order = [("hyenadna", "human_enhancers_cohn"), ("dnabert2", "human_enhancers_cohn"), ("hyenadna", "nt_enhancers"),
             ("dnabert2", "nt_enhancers"), ("hyenadna", "drosophila_enhancers_stark")]
    for mdl, ds in order:
        r = one(pt, model=mdl, dataset=ds)
        degen = ud[(ud.model == mdl) & (ud.dataset == ds) & (ud.train_arm == "comp_equalized")].degenerate.all()
        rows.append([mname[mdl], short(ds), f(r.A_auroc_mean), f"{f(r.B_auroc_mean)} {ci(r.B_auroc_lo, r.B_auroc_hi)}", f(r.C_auroc_mean),
                     f(r.D_auroc_mean) + ("$^\\ast$" if degen else ""), f(r.E_auroc_mean),
                     f"{f(r.S_est, 3, True)} {ci(r.S_lo, r.S_hi, 3, True)}", f"{f(r.T_est, 3, True)} {ci(r.T_lo, r.T_hi, 3, True)}",
                     f"{pct(r.frac_lost)} {pci(r.frac_lost_lo, r.frac_lost_hi)} ({pct(r.frac_lost_random)})"])
    write_table("tab_ft.tex", "tab:ft",
                "End-to-end fine-tuning (seed means over 42--44): the same weights scored on original ($A$), composition-equalized ($B$) "
                "and size-matched random ($C$) test negatives; $D$/$E$: retrained on equalized/random negatives.",
                ["fm_finetune_v2_paper_table.csv", "fm_finetune_v2_unit_diagnostics.csv"],
                "llllllllll",
                ["Model", "Data", "$A$", "$B$ [CI]", "$C$", "$D$", "$E$", "$S{=}A{-}B$ [CI]", "$T{=}C{-}B$ [CI]", "Share lost [CI] (random)"],
                rows, size="\\scriptsize", tabcolsep="1.8pt",
                notes="AUROC; CIs from 1000 resamples of the original test sequences, jointly across arms and averaged over seeds. Share lost "
                      "$=(A{-}B)/(A{-}0.5)$, the share of above-chance AUROC lost (post hoc re-expression of the pre-specified $S$); random share "
                      "$=(A{-}C)/(A{-}0.5)$, in parentheses. $^\\ast$ all three retrained units are degenerate one-class fits (every test probability "
                      "$\\ge$0.5), so this $D$ is not evidence that the equalized arm is unlearnable. Rule verdicts are in Table~\\ref{tab:ftrules}.")
    hc = one(pt, model="hyenadna", dataset="human_enhancers_cohn")
    dc = one(pt, model="dnabert2", dataset="human_enhancers_cohn")
    hd = one(pt, model="hyenadna", dataset="drosophila_enhancers_stark")
    macro("smFTHA", f(hc.A_auroc_mean), "fm_finetune_v2_paper_table.csv hyenadna cohn A")
    macro("smFTHB", f"{f(hc.B_auroc_mean)} {ci(hc.B_auroc_lo, hc.B_auroc_hi)}", "same, B")
    macro("smFTHC", f(hc.C_auroc_mean), "same, C")
    macro("smFTHShare", pct(hc.frac_lost), "same, frac_lost")
    macro("smFTDA", f(dc.A_auroc_mean), "fm_finetune_v2_paper_table.csv dnabert2 cohn A")
    macro("smFTDB", f"{f(dc.B_auroc_mean)} {ci(dc.B_auroc_lo, dc.B_auroc_hi)}", "same, B")
    macro("smFTDC", f(dc.C_auroc_mean), "same, C")
    macro("smFTDShare", pct(dc.frac_lost), "same, frac_lost")
    macro("smFTCtlShare", f"{pct(hd.frac_lost)} {pci(hd.frac_lost_lo, hd.frac_lost_hi)}", "fm_finetune_v2_paper_table.csv hyenadna drosophila frac_lost")
    macro("smFTHParams", intc(hc.n_params_trainable), "same, n_params_trainable")
    macro("smFTHParamsM", f"{hc.n_params_trainable / 1e6:.2f}M", "derived from n_params_trainable")
    macro("smFTDParams", f"{intc(dc.n_params_trainable)} of {intc(dc.n_params_total)}", "fm_finetune_v2_paper_table.csv dnabert2 n_params")
    macro("smFTDParamsPct", pct(dc.n_params_trainable / dc.n_params_total, 2), "derived")
    macro("smFTBalAcc", f(hc.A_bal_acc), "fm_finetune_v2_paper_table.csv hyenadna cohn A_bal_acc")
    macro("smFTNtMCC", f(one(pt, model="hyenadna", dataset="nt_enhancers").A_mcc), "fm_finetune_v2_paper_table.csv hyenadna nt A_mcc")
    for mdl, nm in [("hyenadna", "H"), ("dnabert2", "D")]:
        r = one(pt, model=mdl, dataset="nt_enhancers")
        macro(f"smFTNt{nm}", f"{f(r.A_auroc_mean)}$\\to${f(r.B_auroc_mean)}", f"fm_finetune_v2_paper_table.csv {mdl} nt A->B")

    rows = []
    rl = {"rule1_rides_shortcut": "1: rides the shortcut", "rule2_drop_not_test_imbalance": "2: drop not a test-imbalance artifact",
          "rule3_collapse_cause": "3: cause of the retrained collapse", "rule5_aligned_control_not_destroyed": "5: control not destroyed"}
    for mdl, ds in order:
        r = one(pt, model=mdl, dataset=ds)
        cells = [mname[mdl], short(ds)]
        for k in rl:
            v = r[k]
            cells.append("--" if isnan(v) else ("met" if v is True or str(v) == "True" else ("not met" if v is False or str(v) == "False" else esc(v))))
        rows.append(cells)
    # rule 5 (eval_ft_v2.py): the GC-balanced control is "not destroyed" if B and D stay above chance (CI lower bounds > 0.5)
    # and the control's S lies below the S of every contaminated set of the same model (CI upper bound < that set's CI lower
    # bound). Re-derive the verdict from the table's own columns and name the set whose S the control's S overlaps.
    r5 = one(pt, model="hyenadna", dataset="drosophila_enhancers_stark")
    r5_others = str(r5.rule5_compared_to).split(",")
    r5_rows = {o: one(pt, model="hyenadna", dataset=o) for o in r5_others}
    r5_overlap = [o for o, q in r5_rows.items() if not r5.S_hi < q.S_lo]
    r5_derived = bool(r5.B_auroc_lo > 0.5 and r5.D_auroc_lo > 0.5 and not r5_overlap)
    assert str(r5.rule5_aligned_control_not_destroyed) == str(r5_derived) == "False", (r5.rule5_aligned_control_not_destroyed, r5_derived)
    assert r5.B_auroc_lo > 0.5 and r5.D_auroc_lo > 0.5 and r5_overlap == ["nt_enhancers"], r5_overlap
    r5_nt = r5_rows["nt_enhancers"]
    macro("smFTCtlS", f"{f(r5.S_est, 3, True)} {ci(r5.S_lo, r5.S_hi, 3, True)}", "fm_finetune_v2_paper_table.csv hyenadna drosophila S_est/S_lo/S_hi")
    macro("smFTNtS", f"{f(r5_nt.S_est, 3, True)} {ci(r5_nt.S_lo, r5_nt.S_hi, 3, True)}", "fm_finetune_v2_paper_table.csv hyenadna nt_enhancers S_est/S_lo/S_hi")
    write_table("tab_ftrules.tex", "tab:ftrules",
                "Pre-specified fine-tuning decision rules, as evaluated.",
                ["fm_finetune_v2_paper_table.csv"],
                "ll>{\\raggedright\\arraybackslash}p{0.11\\linewidth}>{\\raggedright\\arraybackslash}p{0.11\\linewidth}>{\\raggedright\\arraybackslash}p{0.22\\linewidth}>{\\raggedright\\arraybackslash}p{0.1\\linewidth}",
                ["Model", "Data", rl["rule1_rides_shortcut"], rl["rule2_drop_not_test_imbalance"], rl["rule3_collapse_cause"], rl["rule5_aligned_control_not_destroyed"]],
                rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Rule 1: $S{>}0$ (CI excludes 0) and either $B$'s CI covers 0.5 or $B{<}0.6$; for HyenaDNA it is met only "
                      "through $B{<}0.6$ ($B$'s CI excludes 0.5). Rule 2: $T{>}0$ (CI excludes 0) and $C$'s CI overlaps $A$'s. Rule 3 reads "
                      "$U{=}E{-}D$ and $R$; for HyenaDNA on \\task{cohn} it rests on degenerate retrains (Table~\\ref{tab:ft}). Rule 5 "
                      "(GC-balanced control not destroyed): $B$ and $D$ stay above chance (CIs above 0.5) and the control's $S$ lies below "
                      "that of each contaminated set (its CI upper bound below the other CI's lower bound). Not met: $B$ and $D$ stay above "
                      f"chance, but the control's $S$, {f(r5.S_est, 3, True)} {ci(r5.S_lo, r5.S_hi, 3, True)}, overlaps that of "
                      f"\\task{{nt\\_enhancers}}, {f(r5_nt.S_est, 3, True)} {ci(r5_nt.S_lo, r5_nt.S_hi, 3, True)}. "
                      "For \\task{nt\\_enhancers} the retrain-side reading is not used, because the "
                      "matching test there is not supported (Table~\\ref{tab:psmdecision}).")

    # ---- paired comparisons across model families on cohn
    cp = load("fm_finetune_v2_compare_paired.csv")
    comp = load("fm_finetune_v2_compare.csv")
    mc = load("fm_finetune_v2_model_contrast.csv")
    fam = {"kmer_lgbm": "$k$-mer LightGBM", "hyenadna_frozen_sep": "HyenaDNA frozen [SEP]", "hyenadna_frozen_mean": "HyenaDNA frozen mean",
           "dnabert2_frozen_mean": "DNABERT-2 frozen mean", "dnabert2_frozen_cls": "DNABERT-2 frozen [CLS]",
           "hyenadna": "HyenaDNA fine-tuned (full)", "dnabert2": "DNABERT-2 fine-tuned (LoRA)"}
    rows = []
    for m in ["kmer_lgbm", "hyenadna_frozen_sep", "hyenadna_frozen_mean", "hyenadna", "dnabert2_frozen_mean", "dnabert2_frozen_cls", "dnabert2"]:
        r = one(comp, dataset="human_enhancers_cohn", model=m)
        rows.append([fam[m], f(r.A_auroc_mean), f"{f(r.B_auroc_mean)} {ci(r.B_auroc_lo, r.B_auroc_hi)}", f(r.C_auroc_mean), f(r.S_est, 3, True),
                     f"{pct(r.frac_lost)} {pci(r.frac_lost_lo, r.frac_lost_hi)}"])
    rows.append("midrule")
    for ft, base in [("hyenadna", "kmer_lgbm"), ("hyenadna", "hyenadna_frozen_sep"), ("hyenadna", "hyenadna_frozen_mean"),
                     ("dnabert2", "dnabert2_frozen_mean"), ("dnabert2", "kmer_lgbm")]:
        r = one(cp, dataset="human_enhancers_cohn", finetuned=ft, baseline=base, quantity="frac_lost")
        rows.append([f"{fam[ft]} $-$ {fam[base]}", "", "", "", "", f"{f(r.difference, 3, True)} {ci(r.ci_lo, r.ci_hi, 3, True)}"])
    r = one(mc, dataset="human_enhancers_cohn", contrast=mc[mc.dataset == "human_enhancers_cohn"].contrast.iloc[0])
    write_table("tab_ftcompare.tex", "tab:ftcompare",
                "One design, three model families on \\task{cohn}: the same test sets for the $k$-mer model, the frozen probes and the "
                "fine-tuned FMs (post hoc, descriptive).",
                ["fm_finetune_v2_compare.csv", "fm_finetune_v2_compare_paired.csv"], "lllllr",
                ["Model", "$A$", "$B$ [CI]", "$C$", "$S$", "Share lost [CI]"], rows, size="\\footnotesize",
                notes="Lower block: paired differences of the share lost (same resamples). The $k$-mer share here is in the fine-tuning design; "
                      "the overlap-weighting design of Section~\\ref{sec:s4} gives a different $k$-mer share, so the design is always named. Non-fine-tuned "
                      "baselines are single deterministic fits on the full training arm.")
    kr = one(comp, dataset="human_enhancers_cohn", model="kmer_lgbm")
    macro("smFTKmerShare", pct(kr.frac_lost), "fm_finetune_v2_compare.csv cohn kmer_lgbm frac_lost")
    r = one(cp, dataset="human_enhancers_cohn", finetuned="hyenadna", baseline="kmer_lgbm", quantity="frac_lost")
    macro("smFTHvsKmer", f"{f(r.difference, 3, True)} {ci(r.ci_lo, r.ci_hi, 3, True)}", "compare_paired cohn hyenadna-kmer frac_lost")
    fz = lambda m: one(comp, dataset="human_enhancers_cohn", model=m).S_est
    macro("smFloorLo", f(min(fz(m) for m in ["hyenadna", "dnabert2", "hyenadna_frozen_sep", "dnabert2_frozen_mean"]), 2),
          "fm_finetune_v2_compare.csv cohn S_est min over FT and frozen headline probes")
    macro("smFloorHi", f(max(fz(m) for m in ["hyenadna", "dnabert2", "hyenadna_frozen_sep", "dnabert2_frozen_mean"]), 2), "same, max")

    # ---- cCRE enrichment and cCRE-free sensitivity
    ce = load("fm_finetune_v2_ccre_enrichment.csv")
    cs = load("fm_finetune_v2_ccre_sensitivity.csv")
    rows = []
    for ds in ["human_enhancers_cohn", "nt_enhancers"]:
        for feat, fl in [("ccre_any", "any cCRE"), ("ccre_els", "enhancer-like (ELS)")]:
            r = one(ce, dataset=ds, split="test", feature=feat, arm="comp_equalized")
            rows.append([short(ds), fl, f"{intc(r.n_neg_overlap)}/{intc(r.n_neg)} ({pct(r.rate_neg)})", pct(r.rate_vs_random), pct(r.rate_pos), sig2(r.fisher_p_vs_random)])
    write_table("tab_ftccre.tex", "tab:ftccre",
                "Equalized test negatives overlap ENCODE cCREs more often than size-matched random test negatives (post hoc Q3).",
                ["fm_finetune_v2_ccre_enrichment.csv"], "llllll",
                ["Data", "Annotation", "Equalized neg.\\ overlapping", "Random neg.", "Positives", "Fisher $p$"], rows,
                notes=f"The leak-free \\task{{nt\\_enhancers}} test has the same {intc(one(ce, dataset='nt_enhancers', split='test', feature='ccre_any', arm='comp_equalized').n_neg)} equalized negatives. cCRE overlap is itself "
                      "composition-correlated (positives overlap far more often), so removing cCRE-overlapping negatives also removes the "
                      "most positive-like negatives; the cCRE-free shares below are a conservative sensitivity analysis, not the "
                      "`true' composition share.")
    rows = []
    for ds in ["human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfree"]:
        for m in ["hyenadna", "dnabert2", "kmer_lgbm"]:
            r = one(cs, dataset=ds, filter="ccre_any_free", model=m)
            rows.append([esc(ds.replace("human_enhancers_", "")) if m == "hyenadna" else "", fam[m], intc(r.B_n_neg),
                         f"{f(r.B_est)} {ci(r.B_lo, r.B_hi)}", f"{f(r.S_est, 3, True)} {ci(r.S_lo, r.S_hi, 3, True)}",
                         f"{f(r.T_est, 3, True)} {ci(r.T_lo, r.T_hi, 3, True)}", f"{pct(r.frac_lost)} (unfiltered {pct(r.unfiltered_frac_lost)})"])
    write_table("tab_ftccrefree.tex", "tab:ftccrefree",
                "Fine-tuned and $k$-mer models re-scored without cCRE-overlapping test negatives (post hoc Q3).",
                ["fm_finetune_v2_ccre_sensitivity.csv"], "lllllll",
                ["Data", "Model", "$B$ neg.", "$B$ [CI]", "$S$ [CI]", "$T$ [CI]", "Share lost"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes="On \\task{cohn} both $S$ and $T$ exclude 0 for every model. On \\task{nt\\_enhancers} "
                      f"({intc(one(cs, dataset='nt_enhancers', filter='ccre_any_free', model='hyenadna').B_n_neg)} cCRE-free equalized negatives) $S$ stays positive but $T$'s CI includes 0 for both fine-tuned models, so there the loss is no longer "
                      "separable from the size-matched random control; the fine-tuned drop is established for \\task{cohn} only.")
    for m, nm in [("hyenadna", "H"), ("dnabert2", "D")]:
        r = one(cs, dataset="human_enhancers_cohn", filter="ccre_any_free", model=m)
        macro(f"smCcreFT{nm}", pct(r.frac_lost), f"fm_finetune_v2_ccre_sensitivity.csv cohn ccre_any_free {m} frac_lost")
        r = one(cs, dataset="nt_enhancers", filter="ccre_any_free", model=m)
        macro(f"smCcreNtT{nm}", ci(r.T_lo, r.T_hi, 3, True), f"same file nt_enhancers {m} T CI")
    r = one(ce, dataset="human_enhancers_cohn", split="test", feature="ccre_any", arm="comp_equalized")
    macro("smCcreEq", pct(r.rate_neg), "fm_finetune_v2_ccre_enrichment.csv cohn test ccre_any comp_equalized rate_neg")
    macro("smFTNtEqNeg", intc(one(ce, dataset="nt_enhancers", split="test", feature="ccre_any", arm="comp_equalized").n_neg),
          "fm_finetune_v2_ccre_enrichment.csv nt_enhancers test ccre_any comp_equalized n_neg (released equalized split)")
    macro("smCcreRand", pct(r.rate_vs_random), "same, rate_vs_random")

    # ---- random-draw reference and leak-free / homology-clean fine-tuning
    rd = load("fm_finetune_v2_randdraw.csv")
    ph = load("fm_finetune_v2_posthoc_paper_table.csv")
    hf = load("homology_recluster_ft_contrasts.csv")
    hman = load("homology_recluster_ft_split_manifest.csv")
    htb = load("homology_recluster_table.csv").set_index("task")
    ntdup = int(one(load("psm_sensitivity_leakage_check.csv"), dataset="nt_enhancers").n_test_verbatim_in_train)
    ccl = one(load("fm_finetune_v2_ccre_enrichment.csv"), dataset="nt_enhancers_leakfree", split="test", feature="ccre_any", arm="comp_equalized")
    rows = []
    for ds in ["human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfree", "drosophila_enhancers_stark"]:
        for m in ["hyenadna", "dnabert2"]:
            q = rd[(rd.dataset == ds) & (rd.model == m)]
            if not len(q):
                continue
            r = q.iloc[0]
            rows.append([esc(ds.replace("human_enhancers_", "").replace("_enhancers_stark", "")), fam[m], intc(r.k_neg), f(r.B),
                         f"{f(r.draw_min)} ({f(r.draw_p2_5)}--{f(r.draw_p97_5)})", f(r.C), intc(r.n_draws_le_B)])
    write_table("tab_ftranddraw.tex", "tab:ftranddraw",
                "Is $B$ a chance draw of a small negative set? 1000 random draws of as many test negatives as the equalized arm holds, "
                "with the same positives (post hoc Q4).",
                ["fm_finetune_v2_randdraw.csv"], "llrrlrr",
                ["Data", "Model", "Neg.\\ per draw", "$B$", "Draw min (2.5--97.5\\%)", "$C$", "Draws $\\le B$"], rows, size="\\footnotesize",
                notes="The random draws do not control for cCRE enrichment (Table~\\ref{tab:ftccre}); for \\task{nt\\_enhancers} read them "
                      "together with Table~\\ref{tab:ftccrefree}. The GC-balanced control's $B$ also lies below every draw.")
    rows = []
    for m in ["hyenadna", "dnabert2"]:
        r = one(ph, model=m, dataset="nt_enhancers_leakfree")
        rows.append([fam[m], f"leak-free \\task{{nt\\_enhancers}} ({intc(r.B_n_test - ccl.n_neg)} pos./{intc(ccl.n_neg)} neg.)", f(r.A_auroc_mean), f"{f(r.B_auroc_mean)} {ci(r.B_auroc_lo, r.B_auroc_hi)}", f(r.C_auroc_mean),
                     f"{f(r.S_est, 3, True)} {ci(r.S_lo, r.S_hi, 3, True)}"])
    hq = lambda ds, c: one(hf, model="hyenadna", dataset=ds, contrast=c)
    rows.append("midrule")
    for ds in ["human_enhancers_cohn", "drosophila_enhancers_stark"]:
        a, b, c_, s = hq(ds, "A_orig_orig"), hq(ds, "B_orig_compeq"), hq(ds, "C_orig_random"), hq(ds, "S_shortcut_worth")
        nneg = one(hman, dataset=ds, split="test", arm="comp_equalized").n_neg
        dlab = "GC-balanced control" if ds == "drosophila_enhancers_stark" else short(ds)
        rows.append([fam["hyenadna"], f"{dlab} on homology-clean data (id80; {intc(nneg)} eq.\\ test neg.)", f(a.estimate), f"{f(b.estimate)} {ci(b.ci_lo, b.ci_hi)}", f(c_.estimate),
                     f"{f(s.estimate, 3, True)} {ci(s.ci_lo, s.ci_hi, 3, True)}"])
    sg = hq("nt_enhancers", "S_gc")
    nneg = one(hman, dataset="nt_enhancers", split="test", arm="comp_equalized").n_neg
    rows.append([fam["hyenadna"], f"\\task{{nt\\_enhancers}} on homology-clean data ({intc(nneg)} eq.\\ test neg.)", f(hq('nt_enhancers', 'A_orig_orig').estimate), "n/e", "n/e",
                 f"$S_\\mathrm{{gc}}$ {f(sg.estimate, 3, True)} {ci(sg.ci_lo, sg.ci_hi, 3, True)}"])
    write_table("tab_ftposthoc.tex", "tab:ftposthoc",
                "Fine-tuning on leak-free and on homology-clean data (post hoc; no decision rule attached).",
                ["fm_finetune_v2_posthoc_paper_table.csv", "homology_recluster_ft_contrasts.csv", "homology_recluster_ft_split_manifest.csv"],
                ">{\\raggedright\\arraybackslash}p{0.2\\linewidth}>{\\raggedright\\arraybackslash}p{0.3\\linewidth}llll", ["Model", "Data", "$A$", "$B$ [CI]", "$C$", "$S$ [CI]"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes=f"Leak-free \\task{{nt\\_enhancers}}: the {intc(ntdup)} test positives that occur verbatim in training are removed; {intc(ccl.n_neg_overlap)} of the {intc(ccl.n_neg)} "
                      f"equalized negatives overlap a cCRE against {intc(round(ccl.rate_vs_random * ccl.n_vs_random))} of {intc(ccl.n_vs_random)} random ones (Table~\\ref{{tab:ftccre}}), so these rows carry that "
                      "caveat. Homology-clean data: the cluster-aware id80 re-split, HyenaDNA only (DNABERT-2 on these splits is packaged, "
                      "not run; Section~\\ref{sec:s9}). n/e: not evaluable (fewer than 10 test negatives in the arm); $S_\\mathrm{gc}$ uses "
                      "the GC-matched arm instead. On this re-split the fine-tuned GC-matched shortcut on \\task{nt\\_enhancers} vanishes; "
                      f"{pct(1 - htb.loc['nt_enhancers'].resplit_id80_frac_test_from_orig_test, 1)} of its test set comes from the provided training split.")
    macro("smHcFTS", f"{f(hq('human_enhancers_cohn', 'S_shortcut_worth').estimate, 3)} {ci(hq('human_enhancers_cohn', 'S_shortcut_worth').ci_lo, hq('human_enhancers_cohn', 'S_shortcut_worth').ci_hi, 3)}", "homology_recluster_ft_contrasts.csv cohn S")
    macro("smHcFTSgc", f"{f(sg.estimate, 3)} {ci(sg.ci_lo, sg.ci_hi, 3, True)}", "homology_recluster_ft_contrasts.csv nt S_gc")

    # ---- GC-matched test effect after fine-tuning
    gp = load("fm_finetune_v2_gc_paired.csv")
    lv = gp[gp.kind == "level"]
    rows = []
    for ds in ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]:
        for m in ["hyenadna_frozen_sep", "hyenadna_frozen_mean", "hyenadna", "dnabert2_frozen_mean", "dnabert2", "kmer_lgbm"]:
            q = lv[(lv.dataset == ds) & (lv.model == m)]
            if not len(q):
                continue
            r = q.iloc[0]
            rows.append([esc(ds.replace("human_enhancers_", "").replace("_enhancers_stark", "")), fam[m], f"{f(r.G_minus_A, 3, True)} {ci(r.ci_lo, r.ci_hi, 3, True)}"])
    write_table("tab_ftgc.tex", "tab:ftgc",
                "GC-matched test effect $G{-}A$ (AUROC of the same weights on the GC-matched test subset minus on the original test; post hoc Q2).",
                ["fm_finetune_v2_gc_paired.csv"], "lll", ["Data", "Model", "$G{-}A$ [CI]"], rows, size="\\footnotesize",
                notes="The frozen rows reproduce the point estimates of the frozen probes' test effect (Table~\\ref{tab:frozen}); "
                      "their CIs come from this analysis's paired bootstrap and differ slightly. After fine-tuning the "
                      "GC-balanced control is no longer flat.")
    r = one(lv, dataset="drosophila_enhancers_stark", model="hyenadna")
    macro("smFTCtlGA", f(r.G_minus_A, 3, True), "fm_finetune_v2_gc_paired.csv drosophila hyenadna G_minus_A")

    # ---- unit diagnostics (validation carve) and runs
    runs = load("fm_finetune_v2_runs.csv")
    ntu = ud[ud.dataset == "nt_enhancers"]
    macro("smValLeakMin", pct(ntu.frac_val_verbatim_in_fit.min()), "fm_finetune_v2_unit_diagnostics.csv nt frac_val_verbatim_in_fit min")
    macro("smValLeakMax", pct(ntu.frac_val_verbatim_in_fit.max()), "same, max")
    for m, nm in [("hyenadna", "H"), ("dnabert2", "D")]:
        q = ntu[ntu.model == m]
        macro(f"smValAuc{nm}", f(q.max_val_auroc.mean(), 3), f"unit_diagnostics nt {m} mean max_val_auroc")
        macro(f"smTestAuc{nm}", f(q.own_arm_test_auroc.mean(), 3), f"unit_diagnostics nt {m} mean own_arm_test_auroc")
    macro("smNunits", intc(len(runs)), "fm_finetune_v2_runs.csv rows")

    copy_fig(RU / "figures" / "fm_finetune_v2_compare.pdf", "fm_finetune_v2_compare.pdf")
    copy_fig(RU / "figures" / "fm_finetune_v2.pdf", "fm_finetune_v2.pdf")


# ============================================================================ S6
def s6():
    pt = load("homology_recluster_paper_table.csv").set_index("task")
    tb = load("homology_recluster_table.csv").set_index("task")
    rows = []
    for t in TASK_ORDER:
        p, q = pt.loc[t], tb.loc[t]
        status = str(q.purge_id80_status)
        if status == "ok":
            purge = f"{f(q.purge_id80_mcc)} / {f(q.purge_id80_std_mcc)}"
            eff = f"{f(q.purge_id80_homology_effect, 3, True)} ({sig2(q.purge_id80_p_null) if q.purge_id80_p_null > 0 else '$<$0.001'})"
            flag = "yes" if bool(q.purge_id80_inflated_prereg) else "no"
        else:
            purge, eff, flag = "n/e", "n/e", "n/e"
        rows.append([short(t), pct(q.red_test_id80, 1), pct(q.exact_dup_test, 1), f(q.original_mcc), purge, eff, flag,
                     f"{f(p.resplit_id80_seedmean)} / {f(p.random_seedmean)}",
                     f"{f(q.resplit_id80_dvr_seedmean, 3, True)} [{cib(q.resplit_id80_dvr_seedmin, 3, True)}, {cib(q.resplit_id80_dvr_seedmax, 3, True)}]",
                     "yes" if bool(q.resplit_id80_inflated_seed_robust) else "no"])
    write_table("tab_hom_tasks.tex", "tab:homtasks",
                "Homology re-clustering (MMseqs2, id80): train/test homology, purge vs.\\ random-removal null, and cluster-aware vs.\\ random re-splits for all 15 tasks.",
                ["homology_recluster_table.csv", "homology_recluster_paper_table.csv"],
                "lrrrlllllr",
                [["", "\\multicolumn{2}{c}{Test sequences}", "", "\\multicolumn{3}{c}{Purge vs.\\ random removal}", "\\multicolumn{3}{c}{Re-split (seeds 42--44)}"],
                 ["Task", "homolog", "exact dup.", "MCC", "MCC raw/adj.", "effect ($p$)", "flag", "cluster/random", "cl.$-$rand.\\ [range]", "robust"]],
                rows, size="\\scriptsize", tabcolsep="2pt",
                notes="Test w/ homolog: test sequences whose connected component contains a training sequence (both strands; $\\ge$80\\% identity "
                      "over $\\ge$80\\% of both sequences). Purge: the same trained model scored on the test split minus those sequences; raw and "
                      "prevalence-adjusted MCC (the purged test set re-weighted so that each class has its original test proportion; "
                      "post hoc, chosen after the first results). Effect vs.\\ null: purged minus the mean of "
                      "1000 random removals with the same per-class counts; Flag: the pre-specified homology-inflation call. For "
                      f"\\task{{nt\\_enhancers}} the purge removes all {intc(one(load('psm_sensitivity_leakage_check.csv'), dataset='nt_enhancers').n_test_verbatim_in_train)} verbatim strong enhancers; against a subtype-matched null the effect is "
                      f"{f(tb.loc['nt_enhancers'].purge_id80_homology_effect_strat, 3, True)} ($p$ = {sig2(tb.loc['nt_enhancers'].purge_id80_p_null_strat)}). "
                      "Re-split: mean over split seeds 42--44; seed-robust = the pre-specified seed-42 call (CI below 0) and cluster $<$ random in "
                      "all three seeds. n/e: a class is eliminated by the purge. The 0.05-MCC materiality cut used in the text is post hoc.")
    evald = tb[tb.purge_id80_status == "ok"]
    n_eval = len(evald)
    n_flag = int(evald.purge_id80_inflated_prereg.astype(bool).sum())
    big = evald[evald.purge_id80_homology_effect.abs() > 0.05]
    macro("smHomMedian", pct(tb.red_test_id80.median(), 1), "homology_recluster_table.csv red_test_id80 median")
    macro("smHomMax", pct(tb.red_test_id80.max(), 1), "same, max")
    macro("smHomNeval", intc(n_eval), "homology_recluster_table.csv purge_id80_status == ok")
    macro("smHomNflag", intc(n_flag), "same, purge_id80_inflated_prereg")
    macro("smHomNbig", intc(len(big)), "same, |purge_id80_homology_effect| > 0.05 (post hoc cut)")
    macro("smHomNontataRaw", f(tb.loc["human_nontata_promoters"].purge_id80_mcc), "homology_recluster_table.csv nontata purge_id80_mcc")
    macro("smHomNontataStd", f(tb.loc["human_nontata_promoters"].purge_id80_std_mcc), "same, purge_id80_std_mcc")
    macro("smHomNontataCl", f(pt.loc["human_nontata_promoters"].resplit_id80_seedmean, 2), "homology_recluster_paper_table.csv nontata resplit_id80_seedmean")
    macro("smHomNontataRa", f(pt.loc["human_nontata_promoters"].random_seedmean, 2), "same, random_seedmean")
    macro("smHomCohn", pct(tb.loc["human_enhancers_cohn"].red_test_id80, 1), "homology_recluster_table.csv cohn red_test_id80")

    # labels vs BIBM Table I
    la = load("homology_recluster_labels_addendum.csv")
    la = la[la.threshold == "id80"].set_index("task")
    rows = []
    n_s42 = n_mean = 0
    for t in TASK_ORDER:
        r, p = la.loc[t], pt.loc[t]
        pub = r.published_submitted
        if pub == "composition":
            purge = s42 = mean = "task type (unchanged by construction)"
            reading = "composition signal preserved"
            same42 = samem = True
        elif str(pub) == "contamination (weak)":
            purge = f"gate fires ({esc(r.sub_preservation)})"
            s42 = mean = "gate fires (CLI: Route 3)"
            reading = "rests on the TATA gate alone"
            same42 = samem = True
        else:
            purge = esc(p.label_after_purge_id80) if isinstance(p.label_after_purge_id80, str) else "n/e"
            s42, mean = esc(r.label_cluster_s42), esc(r.label_cluster_seedmean)
            same42, samem = (r.label_cluster_s42 == pub), (r.label_cluster_seedmean == pub)
            if bool(r.relabel_homology_attributable_seedmean):
                reading = "relabel attributable to homology"
            elif bool(r.relabel_repooling_seedmean):
                reading = "also under random re-splits (re-pooling)"
            else:
                reading = ""
        n_s42 += same42
        n_mean += samem
        rows.append([short(t), esc(pub), purge, s42, mean, esc(r.label_random_seedmean) if pub not in ("composition", "contamination (weak)") else "--",
                     f"{f(r.mcc_cluster_seedmean, 2)}/{f(r.mcc_random_seedmean, 2)}", reading])
    write_table("tab_hom_labels.tex", "tab:homlabels",
                "Route labels after homology control, against the paper's Table~I labels.",
                ["homology_recluster_labels_addendum.csv", "homology_recluster_paper_table.csv"],
                # Task column wide enough for the longest short task id (nt_prom_no_tata) on one line
                ">{\\raggedright\\arraybackslash}p{0.145\\linewidth}>{\\raggedright\\arraybackslash}p{0.09\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}"
                ">{\\raggedright\\arraybackslash}p{0.11\\linewidth}>{\\raggedright\\arraybackslash}p{0.11\\linewidth}>{\\raggedright\\arraybackslash}p{0.09\\linewidth}"
                "l>{\\raggedright\\arraybackslash}p{0.15\\linewidth}",
                ["Task", "Table I", "Purge (prev.-adj.\\ MCC)", "Cluster re-split, seed 42", "Cluster re-split, seed mean", "Random re-split", "MCC cl./rand.", "Reading"],
                rows, size="\\scriptsize", tabcolsep="2.5pt",
                notes=f"Unchanged against Table~I: {n_s42}/15 at seed 42 and {n_mean}/15 by split-seed mean; the purge preserves every label it can score "
                      "on the prevalence-adjusted MCC (\\task{human\\_nontata\\_promoters} would relabel on the raw purged MCC). These counts are "
                      "partly true by construction: the composition labels are task types; once a task fails, contamination vs.\\ shared-motif is "
                      "set by the model-free TATA gate; and the purge re-scores only the MCC and that gate. The informative labels are the solved "
                      "and positional ones. A relabel is attributed to homology only if the random re-split does not reproduce it.")
    macro("smHomLabelsSFortyTwo", f"{n_s42}/15", "derived from homology_recluster_labels_addendum.csv vs submitted labels")
    macro("smHomLabelsMean", f"{n_mean}/15", "same, seed mean")

    # routes 2/3 and FM on homology-clean data
    pos = load("homology_recluster_positional.csv")
    gr = load("homology_recluster_extra_graded.csv")
    iv = load("homology_recluster_interventions.csv")
    xc = load("homology_recluster_extra_cleaning.csv")
    co = load("homology_recluster_extra_coords.csv")
    xs = load("homology_recluster_extra_specificity.csv")
    fm = load("homology_recluster_extra_fm.csv")
    rows = []
    sp = pos[(pos.task == "nt_splice_sites_all") & (pos.n_bins == 8)]
    for dname, dl in [("original", "original (20,001-sequence training arm)"), ("id80", "homology-clean id80"), ("id50", "homology-clean id50")]:
        q = sp[sp.data == dname]
        if len(q):
            r = q.iloc[0]
            rows.append(["Route 2", f"splice $\\Delta$MCC ($B{{=}}8{{-}}B{{=}}1$), {dl}", f"{f(r.delta_mcc, 3, True)} {ci(r.delta_lo, r.delta_hi, 3, True)}"])
    g80 = gr[(gr.task == "nt_splice_sites_all") & (gr.data == "id80")].sort_values("n_bins")
    rows.append(["Route 2", "splice, graded over $B\\in\\{2,4,8,16\\}$ (id80)", ", ".join(f(x, 3, True) for x in g80.delta_mcc)])
    ctl = pos[(pos.data == "id80") & (pos.n_bins == 8) & (pos.role == "non-positional (control)")]
    viol = ctl[(ctl.delta_excludes_0.astype(str) == "True") & (ctl.delta_mcc.abs() > 0.05)]
    rows.append(["Route 2", "selectivity (pre-specified: CI includes 0 or $|\\Delta|\\le$0.05) on the non-positional tasks (id80)",
                 f"{len(ctl) - len(viol)}/{len(ctl)} pass"])
    for _, r in viol.iterrows():
        rows.append(["Route 2", f"selectivity violation: {short(r.task)} (id80; a loss, not a rescue)", f"{f(r.delta_mcc, 3, True)} {ci(r.delta_lo, r.delta_hi, 3, True)}"])
    # homology-clean nt_enhancers equalized arm: test negatives kept, id80 (primary; the paper quotes this value) and id50
    _nq = iv[(iv.experiment.str.startswith("E1")) & (iv.task == "nt_enhancers") & (iv.arm == "comp_equalized")].set_index("data").n_test_neg_kept
    nt_eq80, nt_eq50 = intc(_nq["id80"]), intc(_nq["id50"])
    assert int(_nq["id80"]) < 10 and int(_nq["id50"]) < 10     # below the MCC-feasibility floor at both thresholds
    e1 = iv[(iv.experiment.str.startswith("E1")) & (iv.task == "human_enhancers_cohn")]
    for dname in ["id80"]:
        o = e1[(e1.data == dname) & (e1.arm.str.contains("orig"))]
        e = e1[(e1.data == dname) & (e1.arm.str.contains("equal"))]
        if len(o) and len(e):
            o, e = o.iloc[0], e.iloc[0]
            rows.append(["Route 3", "\\task{cohn} TATA motif-only / GC-content AUROC, original $\\to$ equalized (id80)",
                         f"{f(o.tata_auroc)}$\\to${f(e.tata_auroc)}; {f(o.gc_auroc)}$\\to${f(e.gc_auroc)}"])
    for task in ["human_enhancers_cohn", "nt_enhancers"]:
        a = one(xc, task=task, data="id80", arm="tata_flag")
        b = one(xc, task=task, data="id80", arm="placebo_tata_flag")
        u = one(xc, task=task, data="id80", arm="uncleaned")
        rows.append(["Route 3", f"{short(task)} TATA-flag cleaning vs.\\ size-matched placebo (id80): TATA AUROC; $k$-mer MCC",
                     f"{f(a.tata_auroc)} vs.\\ {f(b.tata_auroc)} (uncleaned {f(u.tata_auroc)}); {f(a.kmer_mcc)} vs.\\ {f(b.kmer_mcc)}"])
    # cohn composition equalization vs. its size-matched random-removal placebo (X1; one matching draw each), with the
    # spread over the 20 equally valid matching draws of the same equalizer on the same homology-clean data (E1 draws)
    e1d = load("homology_recluster_e1_draws_summary.csv")
    xe = one(xc, task="human_enhancers_cohn", data="id80", arm="comp_equalized")
    xp = one(xc, task="human_enhancers_cohn", data="id80", arm="placebo_comp_equalized")
    dr = one(e1d, task="human_enhancers_cohn", data="id80")
    assert int(xe.n_test_neg) == int(xp.n_test_neg) == int(dr.n_test_neg_kept_min) == int(dr.n_test_neg_kept_max)
    assert dr.kmer_auroc_max < xp.kmer_auroc_lo      # every draw lies below the placebo's CI
    e1eq = one(iv[iv.experiment.str.startswith("E1")], task="human_enhancers_cohn", data="id80", arm="comp_equalized")
    rows.append(["Route 3", "\\task{cohn} equalization vs.\\ size-matched placebo (id80; one draw, X1): $k$-mer AUROC",
                 f"{f(xe.kmer_auroc)} {ci(xe.kmer_auroc_lo, xe.kmer_auroc_hi)} vs.\\ {f(xp.kmer_auroc)} {ci(xp.kmer_auroc_lo, xp.kmer_auroc_hi)}; "
                 f"{intc(dr.n_draws)} draws {f(dr.kmer_auroc_min)}--{f(dr.kmer_auroc_max)} (median {f(dr.kmer_auroc_median)})"])
    macro("smHcEqK", f(xe.kmer_auroc), "homology_recluster_extra_cleaning.csv cohn id80 comp_equalized kmer_auroc")
    macro("smHcEqPlacebo", f(xp.kmer_auroc), "same, placebo_comp_equalized kmer_auroc")
    macro("smHcEqDraws", intc(dr.n_draws), "homology_recluster_e1_draws_summary.csv cohn id80 n_draws")
    macro("smHcEqDrawRange", f"{f(dr.kmer_auroc_min)}--{f(dr.kmer_auroc_max)}", "same, kmer_auroc_min/max")
    macro("smHcEqDrawMed", f(dr.kmer_auroc_median), "same, kmer_auroc_median")
    cc = co[(co.task == "human_enhancers_cohn") & (co.data == "id80")]
    if len(cc):
        r = cc.iloc[0]
        rows.append(["Route 3", "\\task{cohn} TATA-flagged negatives at a TSS vs.\\ unflagged (id80)",
                     f"{pct(r.flagged_tss_rate, 1)} vs.\\ {pct(r.unflagged_tss_rate, 1)} ($\\times${f(r.rate_ratio, 2)}; Fisher $p$ = {sig2(r.fisher_p)})"])
    for task in ["human_enhancers_cohn", "nt_enhancers"]:
        q = xs[(xs.task == task) & (xs.data == "id80")]
        if len(q):
            cells = "; ".join(f"{esc(r.motif_class.replace('_', ' '))} {f(r.motif_only_auroc)}" for _, r in q.iterrows())
            rows.append(["Route 3", f"control-motif panel, {short(task)} (id80)", cells])
    # frozen FM on HC data (the paper's cells)
    head_col = "head"
    for model, pool, task, lab in [("hyenadna", "sep", "human_enhancers_cohn", "HyenaDNA [SEP]"), ("hyenadna", "sep", "nt_enhancers", "HyenaDNA [SEP]"),
                                   ("dnabert2", "mean", "human_enhancers_cohn", "DNABERT-2 mean"), ("dnabert2", "mean", "nt_enhancers", "DNABERT-2 mean")]:
        q = fm[(fm.model.str.lower().str.contains(model[:5])) & (fm.pooling == pool) & (fm.task == task) & (fm.data == "id80") & (fm.arm == "gc_match")]
        q = q[q[head_col].astype(str).str.lower().str.contains("lgbm|lightgbm")]
        if len(q):
            r = q.iloc[0]
            rows.append(["FM probe", f"frozen {lab}, {short(task)}: test effect (id80)", f"{f(r.test_effect_auroc, 3, True)} {ci(r.test_effect_auroc_lo, r.test_effect_auroc_hi, 3, True)}"])
    write_table("tab_hom_routes.tex", "tab:homroutes",
                "Routes 2 and 3 and the frozen FM probe on homology-clean data (cluster-aware re-split plus one representative per component and label).",
                ["homology_recluster_positional.csv", "homology_recluster_extra_graded.csv", "homology_recluster_interventions.csv",
                 "homology_recluster_extra_cleaning.csv", "homology_recluster_e1_draws_summary.csv", "homology_recluster_extra_coords.csv",
                 "homology_recluster_extra_specificity.csv", "homology_recluster_extra_fm.csv"],
                "l>{\\raggedright\\arraybackslash}p{0.5\\linewidth}>{\\raggedright\\arraybackslash}p{0.33\\linewidth}",
                ["Route", "Readout", "Value"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes="Equalized-arm TATA and GC-content AUROCs are partly true by construction, so the cleaning-vs.-placebo rows are the less "
                      "circular replication; their $k$-mer MCC compares cleaning with removing as many negatives at random. The equalized "
                      f"$k$-mer AUROC depends on the matching draw: its row shows one draw (X1; equalized TATA AUROC {f(xe.tata_auroc)}; the "
                      f"TATA/GC-content row uses another, E1) and the range over {intc(dr.n_draws)} equally valid draws, all below the placebo's CI. "
                      f"The homology-clean \\task{{nt\\_enhancers}} equalized arm keeps {nt_eq80} test negatives at id80 ({nt_eq50} at id50; "
                      "MCC not evaluable). GC-content AUROC: the AUROC of per-sequence GC content.")
    ra = one(xc, task="human_enhancers_cohn", data="id80", arm="tata_flag")
    rb = one(xc, task="human_enhancers_cohn", data="id80", arm="placebo_tata_flag")
    macro("smHcTataFlag", f(ra.tata_auroc), "homology_recluster_extra_cleaning.csv cohn id80 tata_flag tata_auroc")
    macro("smHcTataPlacebo", f(rb.tata_auroc), "same, placebo_tata_flag")
    rs = sp[sp.data == "id80"].iloc[0]
    ro = sp[sp.data == "original"].iloc[0]
    macro("smHcSplice", f(rs.delta_mcc, 2, True), "homology_recluster_positional.csv splice id80 B8")
    macro("smHcSpliceOrig", f(ro.delta_mcc, 2, True), "same, original")

    au = load("homology_recluster_audit_task.csv")
    au80 = au[au.threshold == "id80"].set_index("task")
    macro("smMissedEnh", intc(au80.loc["human_enhancers_ensembl"].n_exact_missed_by_clusters), "homology_recluster_audit_task.csv enh_ensembl id80 n_exact_missed_by_clusters")
    macro("smMissedWorm", intc(au80.loc["demo_human_or_worm"].n_exact_missed_by_clusters), "same, demo_human_or_worm")
    macro("smMissedEnhLen", f(au80.loc["human_enhancers_ensembl"].missed_len_median, 1), "same, missed_len_median")
    copy_fig(RU / "figures" / "homology_recluster_mcc.pdf", "homology_recluster_mcc.pdf")


# ============================================================================ S7
def s7():
    ag = load("alt_model_agreement.csv")
    lab = load("alt_model_labels.csv")
    r2 = load("alt_model_route2.csv")
    dose = load("alt_model_dose.csv")
    r1 = load("alt_model_route1.csv")
    sc = load("alt_model_route3_sizecontrolled.csv")
    oc = load("alt_model_route3_overlap_contrast.csv")
    rc = load("alt_model_route3_contrast.csv")
    at = load("alt_model_attribution.csv")
    bet = load("psm_sensitivity_between_dataset.csv")
    MN = {"lgbm": "LightGBM", "lr": "LR (MCC-tuned L1/L2)", "ebm": "EBM (128 $k$-mers)", "ebm512": "EBM (512 $k$-mers)", "lra": "LR (AUROC-tuned)"}

    b = ag[(ag.scheme == "bibm") & (ag.bench_arm == "bench")]
    rows = []
    for m in ["lgbm", "lr", "ebm", "ebm512"]:
        r = one(b, model=m)
        rows.append([MN[m], f"{int(r.same_label_as_paper)}/{int(r.n_tasks)}", f"{int(r.failure_labels_same_as_paper)}/{int(r.n_failure_labels_paper)}",
                     esc(r.disagreements_vs_paper) if isinstance(r.disagreements_vs_paper, str) else "none"])
    _lr, _eb = one(b, model="lr"), one(b, model="ebm")
    assert (_lr.same_label_as_paper, _lr.n_tasks, _lr.failure_labels_same_as_paper) == (_eb.same_label_as_paper, _eb.n_tasks, _eb.failure_labels_same_as_paper)
    macro("smAltSame", f"{int(_lr.same_label_as_paper)} of {int(_lr.n_tasks)}", "alt_model_agreement.csv bibm/bench lr (= ebm) same_label_as_paper of n_tasks")
    macro("smAltFailKept", f"{int(_lr.failure_labels_same_as_paper)} of {int(_lr.n_failure_labels_paper)}",
          "alt_model_agreement.csv bibm/bench lr (= ebm) failure_labels_same_as_paper of n_failure_labels_paper")
    write_table("tab_alt_agree.tex", "tab:altagree",
                "Route labels under alternative interpretable classifiers, against the paper's Table~I labels (post hoc mapping).",
                ["alt_model_agreement.csv"], "llll", ["Classifier", "Same label", "Failure-route labels kept", "Changes (task: new label)"], rows,
                notes="Only the classifier is varied; the representation stays the $k$-mer spectrum at LightGBM's validation-selected $k$ "
                      "(the EBM sees the 20-d composition signature plus a train-selected $k$-mer subset). The counts are partly true by "
                      "construction: composition labels are task types, and once a task fails its contamination vs.\\ shared-motif label is set "
                      "by the model-free TATA gate, so a weaker model can hardly change a failure label. The informative checks are the solved "
                      "and positional labels: each change is a solved task that the weaker model fails.")
    rows = []
    L = lab[lab.bench_arm == "bench"]
    for t in TASK_ORDER:
        cells = [short(t), esc(str(one(L, task=t, model="lgbm").paper_bibm).replace("contamination (weak)", "contam. (weak)"))]
        for m in ["lgbm", "lr", "ebm", "ebm512"]:
            q = L[(L.task == t) & (L.model == m)]
            if len(q):
                r = q.iloc[0]
                lbl = esc(str(r.bibm_label).replace("contamination (weak)", "contam. (weak)"))
                if r.bibm_label != r.paper_bibm:
                    lbl = "\\textbf{" + lbl + "}"
                cells.append(f"{lbl} ({f(r.mcc)})")
            else:
                cells.append("--")
        rows.append(cells)
    write_table("tab_alt_labels.tex", "tab:altlabels",
                "Per-task labels (MCC) under each classifier, mapped to the paper's Table~I vocabulary.",
                ["alt_model_labels.csv"], "llllll", ["Task", "Table I", "LightGBM", "LR", "EBM-128", "EBM-512"], rows, size="\\scriptsize", tabcolsep="2pt",
                notes="Bold: a label that differs from Table~I. MCC on the pre-specified capped training arm (tasks capped at 50k training "
                      "sequences); uncapped refits change no label.")

    rows = []
    for m in ["lgbm", "lr", "ebm"]:
        q = one(r2, task="nt_splice_sites_all", model=m, arm="pos")
        oth = r2[(r2.model == m) & (r2.arm == "pos") & (r2.task != "nt_splice_sites_all") & (r2.delta_ci_excludes_0.astype(str) == "True")]
        pooled = one(dose, model=m, motif="POOLED")
        r11 = one(r1, analysis="route1_disc_vs_mcc_n11_uncapped", model=m)
        r9 = one(r1, analysis="route1_disc_vs_mcc_n9_homology_clean_uncapped", model=m)
        pr = one(r1, analysis="presence_null_gb_n9_paper_counts", model=m)
        rows.append([MN[m], f"{f(q.delta_mcc, 3, True)} {ci(q.delta_lo, q.delta_hi, 3, True)}", f(oth.delta_mcc.abs().max(), 3) if len(oth) else "--",
                     f"{f(pooled.slope, 2, True)} {ci(pooled.slope_lo, pooled.slope_hi, 2, True)}",
                     f"{f(r11.rho, 2, True)} ($p$ {sig2(r11.p_perm)})", f"{f(r9.rho, 2, True)} ($p$ {sig2(r9.p_perm)})", f(pr.rho, 2, True)])
    q = one(r2, task="nt_splice_sites_all", model="ebmpos", arm="pos")
    write_table("tab_alt_routes12.tex", "tab:altroutes",
                "Routes 1 and 2 under each classifier.",
                ["alt_model_route2.csv", "alt_model_dose.csv", "alt_model_route1.csv"], "lllllll",
                ["Classifier", "Splice $\\Delta$MCC ($B{=}8{-}1$)", "Largest other $|\\Delta|$", "Dose slope (pooled)", "Real-task $\\rho$, $n{=}11$",
                 "Real-task $\\rho$, $n{=}9^\\dagger$", "Presence $\\rho$ (GB, $n{=}9$)"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes=f"Splice rescue on the same {intc(q.n_train)}-sequence training arm for every model; the EBM with position-binned $k$-mers guaranteed "
                      f"(post hoc) gives {f(q.delta_mcc, 3, True)}. Largest other $|\\Delta|$: largest CI-significant change on the other tasks. "
                      f"Dose slope: semi-synthetic dose-response, {intc(one(dose, model='lgbm', motif='POOLED').n_points)} settings. Real-task $\\rho$: uncapped MCCs where available (the EBM keeps "
                      "its capped MCC on the regulatory task). $^\\dagger$ Without \\task{human\\_nontata\\_promoters} and \\task{nt\\_enhancers}. "
                      "Presence: Spearman of MCC with the number of enriched TFs.")

    rows = []
    tasks6 = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark", "human_enhancers_ensembl", "human_ocr_ensembl", "human_nontata_promoters"]
    rows.append(("group", "AUROC loss beyond size-matched random negatives (random $-$ matched)"))
    for t in tasks6:
        if not len(sc[(sc.task == t) & (sc.feat == "kmer")]):
            continue
        cells = [short(t) + (" (leak-free test)" if t == "nt_enhancers" else "")]
        for m in ["lgbm", "lr", "ebm"]:
            q = sc[(sc.task == t) & (sc.model == m) & (sc.feat == "kmer") & (sc.leakfree_test.astype(str) == ("True" if t == "nt_enhancers" else "False"))]
            cells.append(f"{f(q.iloc[0].comp_effect, 3, True)} {ci(q.iloc[0].comp_effect_lo, q.iloc[0].comp_effect_hi, 3, True)}" if len(q) else "--")
        rows.append(cells)
    rows.append(("group", "Overlap weighting, model fixed: share of above-chance AUROC lost, task $-$ GC-balanced control"))
    for t in tasks6:
        if t == "drosophila_enhancers_stark":
            continue
        cells = [short(t)]
        for m in ["lgbm", "lr", "ebm"]:
            if t == "nt_enhancers":
                cells.append("not interpretable")
                continue
            q = oc[(oc.task == t) & (oc.model == m) & (oc.leakfree_test.astype(str) == "False")]
            cells.append(f"{f(q.iloc[0]['diff'], 3, True)} {ci(q.iloc[0].diff_lo, q.iloc[0].diff_hi, 3, True)}" if len(q) else "--")
        rows.append(cells)
    comp_con = one(bet[bet.kind == 'contrast'], contrast='human_enhancers_cohn - drosophila_enhancers_stark', design='fixed_model_overlap').comp_share_diff
    write_table("tab_alt_route3.tex", "tab:altroute3",
                "Route-3 model-level readouts under each classifier (post hoc).",
                ["alt_model_route3_sizecontrolled.csv", "alt_model_route3_overlap_contrast.csv", "psm_sensitivity_between_dataset.csv"], "llll",
                ["Task", "LightGBM", MN["lr"], MN["ebm"]], rows, size="\\footnotesize",
                notes="Loss beyond random: random-negative AUROC minus AUROC on the released equalized split. Every task tested loses some AUROC "
                      "beyond random negatives, the GC-balanced control included. Overlap contrast: share of above-chance AUROC the unretrained "
                      "model loses under overlap weighting, task minus control; for \\task{nt\\_enhancers} the weights over-correct (weighted "
                      "AUROC below 0.5, share above 1), so its share is not interpretable. The LightGBM contrast for \\task{cohn} is matched by the "
                      f"composition-only model's contrast under the same weights ({f(comp_con, 3, True)}; Table~\\ref{{tab:psmbetween}}), so these "
                      "contrasts measure reliance on the composition gap, not contamination specificity. Under the EBM, which sees composition "
                      "explicitly, the solved gate-positive \\task{human\\_nontata\\_promoters} also loses more than the control.")
    rows = []
    for m in ["lgbm", "lr", "lra", "ebm"]:
        q = one(rc, task="human_enhancers_cohn", model=m, leakfree_test=False)
        rows.append([MN[m], f(q.share_task, 3), f(q.share_control, 3), f"{f(q['diff'], 3, True)} {ci(q.diff_lo, q.diff_hi, 3, True)}"])
    write_table("tab_alt_retrained.tex", "tab:altretrained",
                "Retrained random-negative design: share of above-chance AUROC lost, \\task{cohn} minus the GC-balanced control (post hoc).",
                ["alt_model_route3_contrast.csv"], "llll", ["Classifier", "\\task{cohn}", "Control", "Difference [CI]"], rows,
                notes="The difference excludes 0 only for the MCC-tuned LR; with the AUROC-tuned LR it does not. No model-level readout here is "
                      "contamination-specific.")
    for m, nm in [("lgbm", "L"), ("lr", "R"), ("lra", "A"), ("ebm", "E")]:
        q = one(rc, task="human_enhancers_cohn", model=m, leakfree_test=False)
        macro(f"smRetr{nm}", f"{f(q['diff'], 3, True)} {ci(q.diff_lo, q.diff_hi, 3, True)}", f"alt_model_route3_contrast.csv cohn {m} diff")

    # attribution
    rows = []
    for m in ["lgbm", "lr", "ebm"]:
        q = one(at, task="human_nontata_promoters", model=m)
        ranks = dict(x.split(":") for x in str(q.paper_named_tf_rank).split(";"))
        allm = at[at.model == m]
        rows.append([MN[m], ranks.get("SP1", "--"), ranks.get("SP2", "--"), f(q.jaccard_vs_lgbm, 2) if not isnan(q.jaccard_vs_lgbm) else "--",
                     f(allm.jaccard_vs_lgbm.median(), 2) if m != "lgbm" else "--", f(allm.z_spearman_vs_lgbm.median(), 2) if m != "lgbm" else "--"])
    write_table("tab_alt_attr.tex", "tab:altattr",
                "Motif attribution under each classifier (post hoc): the paper's named TFs on \\task{human\\_nontata\\_promoters}, and agreement "
                "with LightGBM across tasks.",
                ["alt_model_attribution.csv"], "llllll",
                ["Classifier", "SP1 rank", "SP2 rank", "Jaccard vs.\\ LightGBM (nontata)", "Median Jaccard (all tasks)", "Median $z$-profile $\\rho$"], rows,
                notes="Ranks among FDR-enriched vertebrate JASPAR TFs by importance-weighted enrichment; Jaccard: overlap of enriched TF sets.")


# ============================================================================ S8
def s8():
    g = load("gue_results.csv")
    ov = load("gue_overlap_summary.csv").set_index("task")
    au = load("gue_failure_route_audit.csv").set_index("task")
    ndv = ov[ov.independence_class == "near-disjoint"]
    nd_max = ndv.frac_in_any_gb_nt.max()
    mt = ndv[ndv.family == "mouse_tf"]
    nd_n = (mt.frac_in_any_gb_nt * mt.n_released).round()
    rows = []
    for _, r in g.iterrows():
        route = str(r.route).replace(" (no single route)", "").replace(" (motif-discriminable)", "")
        ic = str(ov.loc[r.task].independence_class)
        ic = {"shares with the paper's tasks": "shares with the 15", "near-disjoint": "near-disjoint",
              "shares with NT suite only (not the paper's 15)": "other NT tasks only"}[ic]
        post = ""
        if r.task in au.index:
            a = au.loc[r.task]
            post = "unaffected" if not bool(a.affected_by_posthoc) else ("threshold-dependent" if route == "shared-motif" else "low-signal windows")
        elif route == "hard" and bool(r.hard_only_because_richness):
            post = "hard only via richness threshold"
        cells = [esc(r.task), pct(r.frac_in_any_gb_nt, 1), ic, f"{f(r.mcc)} {ci(r.mcc_lo, r.mcc_hi)}", esc(route),
                 f"{f(r.pos_delta, 3, True)} {ci(r.pos_delta_lo, r.pos_delta_hi, 3, True)}", post]
        rows.append(cells)
    write_table("tab_gue.tex", "tab:gue",
                "The 28 GUE datasets under the unchanged pipeline and route thresholds.",
                ["gue_results.csv", "gue_overlap_summary.csv", "gue_failure_route_audit.csv"],
                "lrllllp{0.2\\linewidth}",
                ["Dataset", "Shared with GB/NT", "Class", "MCC [CI]", "Route", "Positional $\\Delta$MCC [CI]", "Post hoc status"], rows,
                size="\\scriptsize", tabcolsep="3pt",
                notes="Shared: fraction of the dataset's sequences that occur in "
                      "GB or NT tasks as exact matches (either strand; containment in either direction). Class: `shares with the 15', "
                      "shares sequences with the paper's 15 tasks; `other NT tasks only', shares sequences only with NT tasks outside those 15; "
                      f"near-disjoint, at most {pct(nd_max, 1)} shared (mouse TF {intc(nd_n.min())}--{intc(nd_n.max())} sequences per dataset; only covid shares none). "
                      "Route `hard': MCC below 0.7 and none of the three route rules fires (no single route). "
                      "Post hoc status of the failure routes: the yeast (EMP) shared-motif labels depend on a length-dependent richness threshold "
                      "(Table~\\ref{tab:guerule}); the yeast contamination labels fall on measured low-signal genomic windows, not sampled "
                      "negatives (Table~\\ref{tab:gueemp}); only the three positional labels are unaffected. Positional deltas other than the "
                      "three Route-2 tasks are not corrected for multiplicity. covid is 9-class (macro F1 is the GUE metric).")
    counts = g.route.str.replace(" (no single route)", "", regex=False).str.replace(" (motif-discriminable)", "", regex=False).value_counts()
    n_fail = int(counts.get("positional", 0) + counts.get("shared-motif", 0) + counts.get("contamination", 0))
    macro("smGueFail", intc(n_fail), "gue_results.csv route counts (derived)")
    macro("smGueSolved", intc(counts.get("solved", 0)), "same")
    macro("smGueHard", intc(counts.get("hard", 0)), "same")
    macro("smGuePos", intc(counts.get("positional", 0)), "same")
    macro("smGueShared", intc(counts.get("shared-motif", 0)), "same")
    macro("smGueContam", intc(counts.get("contamination", 0)), "same")
    nd = (ov.independence_class == "near-disjoint").sum()
    macro("smGueNear", intc(nd), "gue_overlap_summary.csv independence_class near-disjoint count")
    for t, nm in [("splice_reconstructed", "Splice"), ("prom_core_tata", "Core"), ("prom_300_tata", "ThreeHundred")]:
        r = one(g, task=t)
        macro(f"smGueDelta{nm}", f(r.pos_delta, 3, True), f"gue_results.csv {t} pos_delta")
    ref = load("gue_paper15_reference.csv")
    r = one(ref, task="nt_promoter_tata")
    macro("smNtPromTataDelta", f"{f(r.pos_delta, 3, True)} {ci(r.pos_delta_lo, r.pos_delta_hi, 3, True)}", "gue_paper15_reference.csv nt_promoter_tata pos_delta")
    agree = ref.rule_agrees_with_table1.dropna().astype(str)
    macro("smCliAgree", f"{int((agree == 'True').sum())} of {len(agree)}", "gue_paper15_reference.csv rule_agrees_with_table1")
    hf = ov.loc[[t for t in ov.index if t.startswith("human_tf")]].frac_in_any_gb_nt
    macro("smGueHumanTF", prange(hf.min(), hf.max()), "gue_overlap_summary.csv human_tf frac_in_any_gb_nt range")
    macro("smNearMax", pct(ov[ov.independence_class == "near-disjoint"].frac_in_any_gb_nt.max(), 1), "gue_overlap_summary.csv near-disjoint max frac_in_any_gb_nt")
    macro("smPaperMCCRepro", f"{int(ref.mcc_reproduces_paper.astype(str).eq('True').sum())}", "gue_paper15_reference.csv mcc_reproduces_paper count")

    fs = load("gue_family_summary.csv")
    rows = []
    for _, r in fs.iterrows():
        routes = str(r.routes).replace(" (no single route)", "").replace(" (motif-discriminable)", "")
        fam_ov = ov[ov.family == r.family].frac_in_any_gb_nt
        rows.append([esc(r["name"]), intc(r.n_datasets), esc(r.neg_construction).replace("centre", "center"), prange(fam_ov.min(), fam_ov.max(), 1),
                     esc(routes).replace(";", ","), f(r.mean_kmer, 1), f(r.mean_DNABERT2, 1)])
    write_table("tab_gue_fam.tex", "tab:guefam",
                "GUE task families: how the negatives were built, how much of the DNA is shared with GB/NT, and the labels.",
                ["gue_family_summary.csv"],
                ">{\\raggedright\\arraybackslash}p{0.13\\linewidth}r>{\\raggedright\\arraybackslash}p{0.33\\linewidth}l>{\\raggedright\\arraybackslash}p{0.17\\linewidth}rr",
                ["Family", "$n$", "Negatives / label-0 class (as documented)", "Shared", "Labels", "$k$-mer", "DNABERT-2"], rows,
                size="\\scriptsize", tabcolsep="3pt",
                notes="Shared: range over the family's datasets of the fraction of sequences found in GB/NT. Labels: `hard' = MCC below 0.7 with "
                      "no single route (Table~\\ref{tab:gue}). $k$-mer and DNABERT-2: mean MCC$\\times$100 "
                      "(covid: macro-F1$\\times$100) of this pipeline's $k$-mer LightGBM and of DNABERT-2 as published; the comparison only locates the "
                      "baseline. The yeast `negatives' are the measured low-signal class of each mark, not a sampled negative set.")
    ol0 = ov.frac_in_paper_tasks_label0
    macro("smPromLabelZero", f"{pct(ol0['prom_300_tata'], 1)}/{pct(ol0['prom_core_tata'], 1)}", "gue_overlap_summary.csv frac_in_paper_tasks_label0 prom_300_tata/prom_core_tata")
    fail = g[g.route.isin(["positional", "shared-motif", "contamination"])]
    macro("smGueMinShared", pct(fail.frac_in_any_gb_nt.min(), 1), "gue_results.csv min frac_in_any_gb_nt over failure routes")
    pr = load("gue_predictions.csv")
    rows = [[esc(r.prediction), esc(r.statement).replace("GC AUROC", "GC-content AUROC"), esc(str(r["values"])).replace("; ", ", "), esc(r.outcome)]
            for _, r in pr.iterrows()]
    write_table("tab_gue_pred.tex", "tab:guepred",
                "Predictions stated before the GUE run, and their outcomes.",
                ["gue_predictions.csv"],
                "l>{\\raggedright\\arraybackslash}p{0.36\\linewidth}>{\\raggedright\\arraybackslash}p{0.38\\linewidth}l",
                ["", "Prediction", "Values", "Outcome"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes=f"Of the {intc((pr.outcome != 'not applicable').sum())} applicable predictions (P4 is not applicable), {intc((pr.outcome == 'HOLDS').sum())} held, "
                      f"{intc((pr.outcome == 'PARTLY').sum())} held partly and {intc((pr.outcome == 'FAILS').sum())} failed: Route~2 is not "
                      "confined to splice (P3b), and the mouse-TF composition-only baseline is far from chance (P1a, P1b), because shuffle "
                      "twins were split across train and test (Table~\\ref{tab:guetwins}).")
    oc = pr.outcome.astype(str)
    macro("smGuePredFail", intc((oc == "FAILS").sum()), "gue_predictions.csv outcome FAILS")
    macro("smGuePredApplicable", intc((oc != "not applicable").sum()), "gue_predictions.csv outcome != not applicable")

    rs = load("gue_rule_sensitivity_summary.csv")
    q = rs[rs.suite == "GUE"]
    reg = q[q.variant == "as registered"].iloc[0]
    bd = q[(q.n_hard_to_shared_motif > 0) & (q.n_shared_motif_to_hard > 0)].variant.str.extract(r"(\d+) bp")[0].dropna().astype(int)
    both_dir = f"{bd.min()}--{bd.max()} bp"
    macro("smGueHardRich", intc(reg.n_hard_only_because_richness), "gue_rule_sensitivity_summary.csv GUE as registered n_hard_only_because_richness")
    rows = []
    for _, r in q.iterrows():
        rows.append([esc(r.variant), intc(r.n_changed), intc(r.n_hard_to_shared_motif), intc(r.n_shared_motif_to_hard), esc(r.counts)])
    qq = rs[rs.suite != "GUE"]
    for _, r in qq.iterrows():
        rows.append([esc(f"{r.suite}: {r.variant}"), intc(r.n_changed), intc(r.n_hard_to_shared_motif), intc(r.n_shared_motif_to_hard),
                     f"Table~I agreement {intc(r.table1_agree)}/{intc(r.table1_scorable)}" if not isnan(r.table1_agree) else esc(r.counts)])
    write_table("tab_gue_rule.tex", "tab:guerule",
                "Route-1 richness threshold normalized by sequence length (post hoc PH4): how many labels change.",
                ["gue_rule_sensitivity_summary.csv"],
                ">{\\raggedright\\arraybackslash}p{0.25\\linewidth}rrr>{\\raggedright\\arraybackslash}p{0.42\\linewidth}",
                ["Variant", "Changed", "hard$\\to$shared", "shared$\\to$hard", "Label counts / agreement"], rows, size="\\scriptsize", tabcolsep="3pt",
                notes=f"The absolute richness threshold is unreachable for the shortest GUE tasks, which is why {intc(reg.n_hard_only_because_richness)} of the "
                      f"{intc(reg.n_hard)} GUE `hard' labels are hard; normalizing it moves labels in both directions only at {both_dir}. GB/NT lengths lie near "
                      "the reference length, so their stability is partly mechanical.")

    ep = load("gue_emp_provenance.csv")
    sig = ep[ep.route3_signature.astype(str) == "True"]
    b2 = sig[sig.tata_excess_std_hi < 0.10]
    b4 = sig[sig.tata_excess_std_4strata_hi < 0.10]
    below_txt = (f"With 2 strata the excess falls below the Route-3 threshold (0.10; CI upper bound) for {len(b2)} of {len(sig)} marks "
                 f"({', '.join(esc(x) for x in b2.task) or 'none'}); with 4 strata for {len(b4)}.")
    rows = []
    for _, r in ep[ep.route3_signature.astype(str) == "True"].iterrows():
        rows.append([esc(r.task), esc(r.route), f"{intc(r.gue_n_label1)}/{intc(r.gue_n_label0)}", "yes" if bool(r.counts_match_pham) else "no",
                     f(r.tata_neg_excess_located, 3, True), f"{f(r.tata_excess_std, 3, True)} {ci(r.tata_excess_std_lo, r.tata_excess_std_hi, 3, True)}",
                     pct(r.share_of_excess_explained), f"{f(r.tata_excess_std_4strata, 3, True)}", pct(r.share_of_excess_explained_4strata)])
    write_table("tab_gue_emp.tex", "tab:gueemp",
                "Yeast (EMP) datasets with the Route-3 signature: where the label-0 windows come from, and how much of the TATA excess genomic "
                "location accounts for (post hoc PH5).",
                ["gue_emp_provenance.csv"], "llllllrlr",
                ["Dataset", "Label", "Label 1/0", "= Pham", "Excess", "Location-std.\\ (2 strata) [CI]", "Accounted", "4 strata", "Accounted"], rows,
                size="\\scriptsize", tabcolsep="3pt",
                notes="Label counts equal the positive/negative counts of the Pham et al.\\ 2005 release on every mark, so label 0 is the measured "
                      "low-signal class (low modification relative to H3 for the modification marks), not a sampled negative set. Excess: TATA "
                      "negative-class excess on the uniquely located windows; location-standardized: the same within ORF/intergenic strata "
                      "(2 strata) or finer strata (4). `Accounted' = share of the excess removed by standardization. "
                      f"{below_txt}")
    c = ep[ep.route.astype(str) == "contamination"]
    macro("smEmpShareLo", pct(c.share_of_excess_explained.min()), "gue_emp_provenance.csv contamination marks share_of_excess_explained min")
    macro("smEmpShareHi", pct(c.share_of_excess_explained.max()), "same, max")

    tw = load("gue_twins.csv")
    rows = []
    for _, r in tw[tw.family == "mouse_tf"].iterrows():
        rows.append([esc(r.task), intc(r.n_test), pct(r.frac_test_crosslabel_twin_in_train, 1), f(r.comp_only_auroc_all),
                     f(r.comp_only_auroc_with_crosslabel_twin), f(r.comp_only_auroc_without_crosslabel_twin)])
    write_table("tab_gue_twins.tex", "tab:guetwins",
                "GUE mouse-TF split artifact: dinucleotide-shuffle negatives were split independently of their source positives (post hoc PH2).",
                ["gue_twins.csv"], "lrrrrr",
                ["Dataset", "Test $n$", "Test w/ cross-label twin in train", "Comp.-only AUROC, all", "with twin", "without twin"], rows,
                notes="A test sequence whose shuffle twin (same composition, opposite label) is in the training split inverts a composition-only "
                      "model; on test sequences without such a twin the composition-only AUROC is close to 0.5. When negatives are shuffles of "
                      "the positives, each positive and its shuffle belong in the same split.")
    q = tw[tw.family == "mouse_tf"]
    macro("smTwinAuc", f"{f(q.comp_only_auroc_all.min(), 2)}--{f(q.comp_only_auroc_all.max(), 2)}", "gue_twins.csv mouse_tf comp_only_auroc_all range")

    pc = load("gue_published_comparison.csv")
    if "gap_to_DNABERT2" in pc.columns:
        mcc = pc[pc.metric.astype(str).str.startswith("MCC")]
        gmed = float(mcc.gap_to_DNABERT2.median())
        assert gmed < 0, gmed   # the prose says "below DNABERT-2"
        macro("smGueGapMedian", f(abs(gmed), 1), "gue_published_comparison.csv gap_to_DNABERT2 (k-mer minus DNABERT-2) median over MCC datasets, absolute value (median is negative)")
        macro("smGueGapN", intc(len(mcc)), "same, number of MCC datasets")


# ============================================================================ S9
def s9():
    bundle = ROOT / "external" / "fm_finetune" / "gbkb_ft_gpu_bundle.tar.gz"
    documented = "7e6707d9583c30c74fc6bf735dca57e4ddd81ddcea6f4397c836d421003dd914"
    if bundle.exists():
        h = sha256(bundle)
        assert h == documented, h
        MANIFEST[rel(bundle)] = (h, "")
        bstat = f"\\task{{{h[:16]}}}\\ldots (verified)"
    else:
        bstat = f"\\task{{{documented[:16]}}}\\ldots (documented; bundle not present)"
    for p in ["src/upgrades/fm_finetune/gpu_deferred/jobs.sh", "src/upgrades/homology_recluster/deferred/run_deferred_dnabert_hc.sh",
              "src/upgrades/alt_model/deferred/run_ebm_regulatory_full.sh"]:
        assert (ROOT / p).exists(), p
        register(ROOT / p)
    rows = [
        ["DNABERT-2-117M full-parameter fine-tuning, \\task{cohn} and \\task{nt\\_enhancers} (3 arms $\\times$ seeds 42--44; 18 units)",
         "full-model optimizer state exceeds the laptop's GPU memory", "\\pth{fm_finetune/gpu_deferred/jobs.sh} 2 and 7"],
        ["NT-v2-500M LoRA fine-tuning, \\task{cohn}, \\task{nt\\_enhancers} and the GC-balanced control (27 units)",
         "0.5B parameters; days on the laptop", "\\pth{fm_finetune/gpu_deferred/jobs.sh} 3, 6 and 5"],
        ["DNABERT-2 LoRA fine-tuning on the homology-clean splits (\\task{cohn}, \\task{nt\\_enhancers}; 18 units)",
         "memory above the per-job budget", "\\pth{homology_recluster/deferred/run_deferred_dnabert_hc.sh}"],
        ["Uncapped multiclass EBM on \\task{human\\_ensembl\\_regulatory} (1 unit)", "about 9--10 h at 3 threads; cannot change a label",
         "\\pth{alt_model/deferred/run_ebm_regulatory_full.sh}"],
        ["DNABERT-2 frozen probe and LoRA fine-tuning on the GC-balanced control", "not packaged (the exclusion was a choice; Table~\\ref{tab:d9})",
         "--"],
        ["Seed-pinned re-draw of the dinucleotide-shuffled negatives", "not packaged (single-draw dependence is disclosed instead; Table~\\ref{tab:shuffle})", "--"],
    ]
    write_table("tab_deferred.tex", "tab:deferred",
                "Analyses packaged or recommended but NOT run. No number in the paper or in this supplement comes from them.",
                ["src/upgrades/fm_finetune/gpu_deferred/README.md", "src/upgrades/homology_recluster/deferred/README.md",
                 "src/upgrades/alt_model/deferred/README.md"],
                ">{\\raggedright\\arraybackslash}p{0.44\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}",
                ["Analysis (not run)", "Why not run", "Package / command"], rows, size="\\footnotesize", tabcolsep="3pt",
                notes="GPU bundle for the fine-tuning jobs: \\pth{external/fm_finetune/gbkb_ft_gpu_bundle.tar.gz}, SHA-256 " + bstat +
                      ". Paths are under \\task{src/upgrades/}. After a GPU hand-back, the torch-free scoring chain in "
                      "\\task{src/upgrades/fm\\_finetune/README.md} (step 3) re-scores every unit with the same pre-specified rules.")


# ============================================================================ manifest + macros
def write_macros():
    L = ["% generated by bibm/supplement/make_sm_tables.py -- do not edit"]
    for n, v, s in MACROS:
        L.append(f"\\newcommand{{\\{n}}}{{{v}}}% {s}")
    (OUT / "sm_numbers.tex").write_text("\n".join(L) + "\n")


def write_manifest():
    items = sorted(MANIFEST.items())
    rows = []
    for pth, (h, n) in items:
        rows.append(f"\\task{{{esc(pth)}}} & \\task{{{h[:12]}}} & {intc(n) if n != '' else ''} \\\\")
    # IEEEtran-style caption (centred "TABLE Sn" line, small-caps caption) set by hand: a longtable \caption would use the
    # LaTeX default "TABLE Sn: ..." style, unlike every other table of this supplement
    L = ["% generated by bibm/supplement/make_sm_tables.py -- do not edit",
         "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{4pt}",
         "\\refstepcounter{table}\\label{tab:manifest}",
         "\\begin{center}\\footnotesize TABLE~\\thetable\\\\[1pt]{\\scshape Every input file read by "
         "\\task{make\\_sm\\_tables.py}, with its SHA-256 prefix and row count. Source: this generator.}\\end{center}",
         "\\begin{longtable}{@{}p{0.72\\linewidth}lr@{}}",
         "\\toprule File & SHA-256 (first 12) & Rows \\\\ \\midrule\\endfirsthead",
         "\\toprule File & SHA-256 (first 12) & Rows \\\\ \\midrule\\endhead",
         "\\bottomrule\\endlastfoot"] + rows + ["\\end{longtable}", "\\endgroup"]
    (OUT / "manifest.tex").write_text("\n".join(L) + "\n")


def main():
    s1()
    s2()
    s3()
    s4()
    s5()
    s6()
    s7()
    s8()
    s9()
    write_macros()
    write_manifest()
    print(f"wrote {len(list(OUT.glob('tab_*.tex')))} tables, {len(MACROS)} macros, {len(MANIFEST)} manifest entries -> {rel(OUT)}")


if __name__ == "__main__":
    main()
