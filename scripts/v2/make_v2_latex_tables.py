"""Generate comprehensive LaTeX tables for the v2 experiment.

Reads CSV artefacts from v2_results_second_run/v2/tables/ and writes a set of
self-contained .tex files into report/v2_tables/.  Each file contains a
longtable with all metrics for the corresponding artefact, intended to be
``\\input{}``-ed from report/v2_section.tex.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "v2_results_second_run" / "v2" / "tables"
OUT = ROOT / "report" / "v2_tables"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(x, n=3):
    if pd.isna(x):
        return "---"
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def fmti(x):
    if pd.isna(x):
        return "---"
    return f"{int(x)}"


def write_tex(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}  ({len(text)} chars)")


# ---------------------------------------------------------------------------
# 1. Final tabular results (one row per (T,L,W,algo) winner) -- all metrics
# ---------------------------------------------------------------------------
def make_tabular_full():
    df = pd.read_csv(TABLES / "final_test_v2_tabular.csv")
    df = df.sort_values(
        ["algo", "threshold", "input_length_days", "washout_days"]
    ).reset_index(drop=True)

    lines = [
        r"\begin{longtable}{lcccccccccccc}",
        r"\caption{Full tabular results on the held-out test partition for every"
        r" $(T,L,W,\text{algo})$ winner (81 rows). Metrics at the default $0.5$"
        r" cutoff: accuracy, precision, recall, F1.  Metrics at the"
        r" validation-tuned cutoff $\tau^*$: F1, precision, recall.  Ranking"
        r" metrics: ROC-AUC, PR-AUC, Brier score.  $n_+$ is the test-positive"
        r" count.}\label{tab:v2_tabular_full}\\",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n$ & $n_+$ & Acc & Prec & Rec & F1 & "
        r"$\tau^*$ & F1$_{\tau^*}$ & ROC & PR & Brier \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n$ & $n_+$ & Acc & Prec & Rec & F1 & "
        r"$\tau^*$ & F1$_{\tau^*}$ & ROC & PR & Brier \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        lines.append(
            " & ".join([
                r.algo.upper(),
                cell,
                fmti(r.n_test),
                fmti(r.test_positive),
                fmt(r.test_accuracy),
                fmt(r.test_precision),
                fmt(r.test_recall),
                fmt(r.test_f1),
                fmt(r.tuned_threshold),
                fmt(r.test_f1_tuned),
                fmt(r.test_roc_auc),
                fmt(r.test_pr_auc),
                fmt(r.test_brier),
            ])
            + r" \\"
        )
    lines.append(r"\end{longtable}")
    write_tex(OUT / "tabular_full.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 2. Tabular bootstrap CIs (per (T,L,W,algo)) -- ranking metrics
# ---------------------------------------------------------------------------
def make_tabular_ci():
    df = pd.read_csv(TABLES / "analysis_v2_tabular_ci.csv")
    df = df.sort_values(
        ["algo", "threshold", "input_length_days", "washout_days"]
    ).reset_index(drop=True)
    lines = [
        r"\begin{longtable}{lccccc}",
        r"\caption{Bootstrap $95\%$ confidence intervals (2,000 resamples) on"
        r" test PR-AUC and ROC-AUC for every tabular winner.}"
        r"\label{tab:v2_tabular_ci}\\",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n_+$ & PR-AUC [95\% CI] & ROC-AUC [95\% CI] \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n_+$ & PR-AUC [95\% CI] & ROC-AUC [95\% CI] \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        pr = f"{fmt(r.pr_auc)}\\,[{fmt(r.pr_auc_lo95)},{fmt(r.pr_auc_hi95)}]"
        roc = f"{fmt(r.roc_auc)}\\,[{fmt(r.roc_auc_lo95)},{fmt(r.roc_auc_hi95)}]"
        lines.append(
            f"{r.algo.upper()} & {cell} & {fmti(r.n_test_positive)} & {pr} & {roc} \\\\"
        )
    lines.append(r"\end{longtable}")
    write_tex(OUT / "tabular_ci.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 3. Deep-learning per-seed table (RNN winner) -- all metrics
# ---------------------------------------------------------------------------
def make_dl_seeds():
    df = pd.read_csv(TABLES / "final_test_v2_dl.csv")
    df = df.sort_values("seed").reset_index(drop=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Deep-learning winner (RNN, $(T,L,W)=(3,14,14)$, all sensors)"
        r" refit across five seeds.  Default-cutoff accuracy / precision /"
        r" recall / F1 plus ranking metrics on the held-out test partition.}"
        r"\label{tab:v2_dl_seeds}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"Seed & Best Epoch & Acc & Prec & Rec & F1 & ROC-AUC & PR-AUC & Brier \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            " & ".join([
                fmti(r.seed),
                fmti(r.best_epoch),
                fmt(r.test_accuracy),
                fmt(r.test_precision),
                fmt(r.test_recall),
                fmt(r.test_f1),
                fmt(r.test_roc_auc),
                fmt(r.test_pr_auc),
                fmt(r.test_brier),
            ])
            + r" \\"
        )
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    lines.append(r"\midrule")
    lines.append(
        " & ".join([
            r"\textbf{Mean}",
            r"---",
            fmt(mean.test_accuracy),
            fmt(mean.test_precision),
            fmt(mean.test_recall),
            fmt(mean.test_f1),
            fmt(mean.test_roc_auc),
            fmt(mean.test_pr_auc),
            fmt(mean.test_brier),
        ])
        + r" \\"
    )
    lines.append(
        " & ".join([
            r"\textbf{Std}",
            r"---",
            fmt(std.test_accuracy),
            fmt(std.test_precision),
            fmt(std.test_recall),
            fmt(std.test_f1),
            fmt(std.test_roc_auc),
            fmt(std.test_pr_auc),
            fmt(std.test_brier),
        ])
        + r" \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    write_tex(OUT / "dl_seeds.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 4. Sensor ablation -- ALL 729 rows
# ---------------------------------------------------------------------------
def make_sensor_full():
    df = pd.read_csv(TABLES / "sensor_ablation_v2.csv")
    df = df.sort_values(
        ["algo", "threshold", "input_length_days", "washout_days", "subset"]
    ).reset_index(drop=True)
    lines = [
        r"\begin{longtable}{llcccccccccc}",
        r"\caption{Full sensor-ablation grid: every $(T,L,W,\text{algo},"
        r"\text{subset})$ refit.  Columns report validation and test "
        r"PR-AUC / ROC-AUC / F1 plus the test accuracy.  729 rows.}"
        r"\label{tab:v2_sensor_full}\\",
        r"\toprule",
        r"Algo & Subset & $(T,L,W)$ & $n_{\text{feat}}$ & Val PR & Val ROC & "
        r"Val F1 & Test Acc & Test Prec & Test Rec & Test ROC & Test PR \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Algo & Subset & $(T,L,W)$ & $n_{\text{feat}}$ & Val PR & Val ROC & "
        r"Val F1 & Test Acc & Test Prec & Test Rec & Test ROC & Test PR \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        subset = str(r.subset).replace("_", r"\_")
        lines.append(
            " & ".join([
                r.algo.upper(),
                subset,
                cell,
                fmti(r.n_features),
                fmt(r.val_pr_auc),
                fmt(r.val_roc_auc),
                fmt(r.val_f1),
                fmt(r.test_accuracy),
                fmt(r.test_precision),
                fmt(r.test_recall),
                fmt(r.test_roc_auc),
                fmt(r.test_pr_auc),
            ])
            + r" \\"
        )
    lines.append(r"\end{longtable}")
    write_tex(OUT / "sensor_ablation_full.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 5. Sensor ablation top-K (already pre-computed)
# ---------------------------------------------------------------------------
def make_sensor_topk():
    df = pd.read_csv(TABLES / "analysis_v2_sensor_ablation_topk.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Top-3 sensor subsets per tabular algorithm, ranked by"
        r" validation PR-AUC.  All metrics shown.}"
        r"\label{tab:v2_ablation_topk}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Algo & Subset & $(T,L,W)$ & Val PR & Val ROC & Val F1 & "
        r"Test PR & Test ROC & Test F1 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        subset = str(r.subset).replace("_", r"\_")
        lines.append(
            " & ".join([
                r.algo.upper(),
                subset,
                cell,
                fmt(r.val_pr_auc),
                fmt(r.val_roc_auc),
                fmt(r.val_f1),
                fmt(r.test_pr_auc),
                fmt(r.test_roc_auc),
                fmt(r.test_f1),
            ])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    write_tex(OUT / "sensor_topk.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 6. Leakage probe (81 rows)
# ---------------------------------------------------------------------------
def make_leakage():
    df = pd.read_csv(TABLES / "leakage_probe_v2_summary.csv")
    df = df.sort_values(
        ["algo", "threshold", "input_length_days", "washout_days"]
    ).reset_index(drop=True)
    lines = [
        r"\begin{longtable}{lcccccc}",
        r"\caption{Leakage probe: 5 shuffled-label refits per configuration."
        r"  A leakage-free pipeline should show mean validation ROC-AUC near"
        r" $0.5$.  Overall mean across all 81 configurations is $0.505$"
        r" $(\pm 0.125)$.}\label{tab:v2_leakage}\\",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n_{\text{shuf}}$ & Val ROC-AUC (mean$\pm$std) & "
        r"Val PR-AUC (mean$\pm$std) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Algo & $(T,L,W)$ & $n_{\text{shuf}}$ & Val ROC-AUC (mean$\pm$std) & "
        r"Val PR-AUC (mean$\pm$std) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        roc = f"{fmt(r.val_roc_auc_mean)} $\\pm$ {fmt(r.val_roc_auc_std)}"
        pr = f"{fmt(r.val_pr_auc_mean)} $\\pm$ {fmt(r.val_pr_auc_std)}"
        lines.append(
            f"{r.algo.upper()} & {cell} & {fmti(r.n_shuffles)} & {roc} & {pr} \\\\"
        )
    lines.append(r"\end{longtable}")
    write_tex(OUT / "leakage_full.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 7. v1-vs-v2 comparison (81 rows)
# ---------------------------------------------------------------------------
def make_v1_v2():
    df = pd.read_csv(TABLES / "analysis_v2_v1_vs_v2.csv")
    df = df.sort_values(
        ["algo", "threshold", "input_length_days", "washout_days"]
    ).reset_index(drop=True)
    lines = [
        r"\begin{longtable}{lcccccccc}",
        r"\caption{Per-cell comparison between v1 (questionnaire-event"
        r" episode, no held-out validation, HPO on test) and v2 (event-"
        r"episode, 3-way patient split, HPO on validation only).  $\Delta$"
        r" columns are v2$-$v1.  Cells without a matching v1 entry show"
        r" \texttt{---}.}\label{tab:v2_vs_v1}\\",
        r"\toprule",
        r"Algo & $(T,L,W)$ & v1 PR & v1 ROC & v2 PR [95\% CI] & v2 ROC [95\% CI]"
        r" & $\Delta$PR & $\Delta$ROC \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Algo & $(T,L,W)$ & v1 PR & v1 ROC & v2 PR [95\% CI] & v2 ROC [95\% CI]"
        r" & $\Delta$PR & $\Delta$ROC \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in df.iterrows():
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        v2pr = (
            f"{fmt(r.v2_pr_auc)}\\,[{fmt(r.v2_pr_auc_lo95)},"
            f"{fmt(r.v2_pr_auc_hi95)}]"
        )
        v2roc = (
            f"{fmt(r.v2_roc_auc)}\\,[{fmt(r.v2_roc_auc_lo95)},"
            f"{fmt(r.v2_roc_auc_hi95)}]"
        )
        lines.append(
            " & ".join([
                r.algo.upper(),
                cell,
                fmt(r.v1_best_pr_auc),
                fmt(r.v1_best_roc_auc),
                v2pr,
                v2roc,
                fmt(r.pr_auc_delta),
                fmt(r.roc_auc_delta),
            ])
            + r" \\"
        )
    lines.append(r"\end{longtable}")
    write_tex(OUT / "v1_vs_v2.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 8. Sample counts (27 rows)
# ---------------------------------------------------------------------------
def make_sample_counts():
    df = pd.read_csv(TABLES / "sample_counts_v2.csv")
    df = df.sort_values(
        ["threshold", "input_length_days", "washout_days"]
    ).reset_index(drop=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sample counts for every $(T,L,W)$ cell across the entire"
        r" cohort.  Excluded counts explain how the positive and negative"
        r" pools are pruned.}\label{tab:v2_sample_counts}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$T$ & $L$ & $W$ & Pos cand & Pos kept & Pos exc (hist) & "
        r"Neg cand & Neg kept & Neg exc (total) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            " & ".join([
                fmti(r.threshold),
                fmti(r.input_length_days),
                fmti(r.washout_days),
                fmti(r.positive_candidates),
                fmti(r.positive_samples),
                fmti(r.positive_excluded_not_enough_history),
                fmti(r.negative_candidates),
                fmti(r.negative_samples),
                fmti(r.negative_excluded_total),
            ])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    write_tex(OUT / "sample_counts.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 9. Headline summary (one row per family)
# ---------------------------------------------------------------------------
def make_headline():
    summ = pd.read_csv(TABLES / "final_test_v2_summary.csv")
    ci = pd.read_csv(TABLES / "analysis_v2_tabular_ci.csv")
    dl_ci = pd.read_csv(TABLES / "analysis_v2_dl_ci.csv")

    def ci_for(algo, t, l, w):
        m = ci[
            (ci.algo == algo)
            & (ci.threshold == t)
            & (ci.input_length_days == l)
            & (ci.washout_days == w)
        ]
        if len(m):
            r = m.iloc[0]
            return r.pr_auc_lo95, r.pr_auc_hi95, r.roc_auc_lo95, r.roc_auc_hi95
        return None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{v2 per-family headline winners on the held-out test set."
        r" Tabular rows show the single deterministic fit; the DL row reports"
        r" mean$\pm$std across five seeds.  All eight metrics shown:"
        r" accuracy, default-cutoff F1, validation-tuned F1, precision,"
        r" recall, ROC-AUC (with 95\% bootstrap CI), PR-AUC (with 95\% CI),"
        r" Brier score.}\label{tab:v2_headline}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Family & Model & $(T,L,W)$ & Acc & F1 & F1$_{\tau^*}$ & "
        r"Prec$_{\tau^*}$ & Rec$_{\tau^*}$ & ROC-AUC [95\% CI] & "
        r"PR-AUC [95\% CI] & Brier \\",
        r"\midrule",
    ]
    tab = summ[summ.family == "tabular"]
    for _, r in tab.iterrows():
        # Need to fetch per-row accuracy / precision tuned from final_test_v2_tabular
        full = pd.read_csv(TABLES / "final_test_v2_tabular.csv")
        m = full[
            (full.algo == r["name"])
            & (full.threshold == r.threshold)
            & (full.input_length_days == r.input_length_days)
            & (full.washout_days == r.washout_days)
        ].iloc[0]
        c = ci_for(r["name"], r.threshold, r.input_length_days, r.washout_days)
        roc_ci = (
            f"{fmt(r.test_roc_auc)}\\,[{fmt(c[2])},{fmt(c[3])}]" if c else fmt(r.test_roc_auc)
        )
        pr_ci = (
            f"{fmt(r.test_pr_auc)}\\,[{fmt(c[0])},{fmt(c[1])}]" if c else fmt(r.test_pr_auc)
        )
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        lines.append(
            " & ".join([
                "Tabular",
                r["name"].upper(),
                cell,
                fmt(m.test_accuracy),
                fmt(r.test_f1),
                fmt(r.test_f1_tuned),
                fmt(m.test_precision_tuned),
                fmt(m.test_recall_tuned),
                roc_ci,
                pr_ci,
                fmt(r.test_brier),
            ])
            + r" \\"
        )
    # DL row
    dl = summ[summ.family == "deep_learning"]
    if len(dl):
        r = dl.iloc[0]
        # use multiseed_summary for richer metrics
        ms = pd.read_csv(TABLES / "tune_dl_v2_multiseed_summary.csv").iloc[0]
        dlci = dl_ci.iloc[0] if len(dl_ci) else None
        cell = (
            f"({int(r.threshold)},{int(r.input_length_days)},"
            f"{int(r.washout_days)})"
        )
        roc_str = (
            f"{fmt(ms.test_roc_auc_mean)} $\\pm$ {fmt(ms.test_roc_auc_std)}"
        )
        pr_str = (
            f"{fmt(ms.test_pr_auc_mean)} $\\pm$ {fmt(ms.test_pr_auc_std)}"
        )
        lines.append(
            " & ".join([
                "Deep",
                f"{r['name'].upper()} (5 seeds)",
                cell,
                f"{fmt(ms.test_accuracy_mean)}",
                f"{fmt(ms.test_f1_mean)}",
                "---",
                "---",
                "---",
                roc_str,
                pr_str,
                f"{fmt(ms.test_brier_mean)}",
            ])
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    write_tex(OUT / "headline.tex", "\n".join(lines) + "\n")


def main():
    make_headline()
    make_tabular_full()
    make_tabular_ci()
    make_dl_seeds()
    make_sensor_full()
    make_sensor_topk()
    make_leakage()
    make_v1_v2()
    make_sample_counts()


if __name__ == "__main__":
    main()
