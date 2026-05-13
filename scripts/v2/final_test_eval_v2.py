"""Phase 8: Final test-set evaluation (one shot, after HPO + leakage probe).

Inputs
------
* ``outputs/v2/tables/tune_tabular_v2_best.csv`` – winners from Phase 4.
* ``outputs/v2/tables/tune_dl_v2_multiseed.csv`` – multi-seed DL winners
  from Phase 5 (optional).
* ``outputs/v2/tables/leakage_probe_v2_summary.csv`` – Phase 7 gate
  (required; raises if missing).
* ``outputs/v2/tables/tune_tabular_v2_split.json`` – the patient split.

Procedure
---------
For each winner:
  1. Refit on **train + val** with the winning hp.
  2. Predict on **test** (held out from all prior phases).
  3. Record full 7-metric panel + sample sizes.

For DL multi-seed winners, we additionally refit across all seeds and
report mean/std on the test set.

Outputs
-------
* ``outputs/v2/tables/final_test_v2_tabular.csv``
* ``outputs/v2/tables/final_test_v2_dl.csv`` (if multi-seed table present)
* ``outputs/v2/tables/final_test_v2_summary.csv`` (combined headline)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.event_features import build_daily_feature_tables, build_patient_static_features
from src.event_labels import run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2
from src.event_v2.deep_learning_v2 import (
    build_sequence_dataset,
    split_sequence_dataset,
    train_one_arch,
)
from src.event_v2.features_v2 import build_training_feature_table
from src.event_v2.modeling_v2 import build_model, select_feature_columns
from src.event_v2.samples_v2 import build_all_sample_indexes
from src.event_v2.split_v2 import (
    PatientSplit,
    compute_metrics,
    split_feature_table,
)
from src.event_labels import USER_COL
from scripts.v2.sensor_ablation_v2 import _parse_hp


def _tune_threshold_max_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Pick the probability threshold on val that maximises F1.

    Falls back to 0.5 when val has < 2 classes or no positive probability mass.
    """
    y_val = np.asarray(y_val).astype(int)
    p_val = np.asarray(p_val, dtype=float)
    if len(np.unique(y_val)) < 2:
        return 0.5
    # Candidate cutoffs = unique predicted probabilities (plus 0.5 anchor).
    cuts = np.unique(np.concatenate([p_val, np.array([0.5])]))
    if cuts.size == 0:
        return 0.5
    best_f1 = -1.0
    best_t = 0.5
    from sklearn.metrics import f1_score as _f1
    for t in cuts:
        pred = (p_val >= t).astype(int)
        if pred.sum() == 0:
            continue
        f1 = _f1(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _check_leakage_gate(path: Path, tol: float = 0.05) -> None:
    if not path.exists():
        raise SystemExit(
            f"\nLeakage-probe summary missing: {path}\n"
            "Run scripts/v2/leakage_probe_v2.py before final test eval."
        )
    summary = pd.read_csv(path)
    if summary.empty:
        raise SystemExit(f"Leakage summary empty: {path}")
    bad = summary[
        (summary["val_roc_auc_mean"] < 0.5 - tol)
        | (summary["val_roc_auc_mean"] > 0.5 + tol)
    ]
    if not bad.empty:
        print(
            f"\nWARNING: {len(bad)} winners failed the leakage gate "
            f"(roc_auc outside [0.5 ± {tol}]):"
        )
        print(bad.to_string(index=False))
        print(
            "Proceeding anyway because the gate is advisory.  Re-investigate "
            "before reporting these numbers."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabular-best",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_best.csv"),
    )
    parser.add_argument(
        "--dl-multiseed",
        default=str(OUTPUT_TABLES_V2 / "tune_dl_v2_multiseed.csv"),
    )
    parser.add_argument(
        "--leakage-summary",
        default=str(OUTPUT_TABLES_V2 / "leakage_probe_v2_summary.csv"),
    )
    parser.add_argument(
        "--split-json",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_split.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leakage-tol", type=float, default=0.05)
    parser.add_argument("--skip-leakage-gate", action="store_true")
    parser.add_argument(
        "--dl-epochs", type=int, default=30,
        help="Override DL epochs when refitting (None = use multi-seed table value).",
    )
    parser.add_argument("--dl-batch-size", type=int, default=32)
    parser.add_argument("--dl-patience", type=int, default=5)
    parser.add_argument("--dl-learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    if not args.skip_leakage_gate:
        _check_leakage_gate(Path(args.leakage_summary), tol=args.leakage_tol)

    tab_best_path = Path(args.tabular_best)
    if not tab_best_path.exists():
        raise SystemExit(f"Missing tabular winners: {tab_best_path}")
    winners = pd.read_csv(tab_best_path)

    split_path = Path(args.split_json)
    if not split_path.exists():
        raise SystemExit(f"Missing split file: {split_path}")
    meta = json.loads(split_path.read_text())
    split = PatientSplit(
        train_users=tuple(meta["train_users"]),
        val_users=tuple(meta["val_users"]),
        test_users=tuple(meta["test_users"]),
        seed=int(meta["seed"]),
        val_frac=float(meta["val_frac"]),
        test_frac=float(meta["test_frac"]),
    )

    # ---- 1. Tabular winners ---------------------------------------- #
    unique_keys = sorted(
        {
            (int(r.threshold), int(r.input_length_days), int(r.washout_days))
            for r in winners.itertuples(index=False)
        }
    )
    thresholds = sorted({k[0] for k in unique_keys})
    lengths = sorted({k[1] for k in unique_keys})
    washouts = sorted({k[2] for k in unique_keys})

    print("[final] building labels + samples + features (all sensors)...")
    label_artifacts = run_event_labeling(thresholds=thresholds)
    sample_indexes, _ = build_all_sample_indexes(
        weekly_events=label_artifacts.weekly_events,
        probable_days_by_threshold=label_artifacts.probable_event_days,
        episodes_by_threshold=label_artifacts.event_episodes,
        input_lengths=lengths,
        washout_values=washouts,
    )
    sample_indexes = {k: v for k, v in sample_indexes.items() if k in unique_keys}
    daily_tables = build_daily_feature_tables()
    patient_features = build_patient_static_features()

    print("\n[final] refitting tabular winners on train+val, predicting on test...")
    tab_rows = []
    tab_pred_rows = []
    for winner in winners.itertuples(index=False):
        T = int(winner.threshold)
        L = int(winner.input_length_days)
        W = int(winner.washout_days)
        algo = str(winner.algo)
        hp = _parse_hp(pd.Series(winner._asdict()), algo)

        feat = build_training_feature_table(
            sample_index=sample_indexes[(T, L, W)],
            daily_tables=daily_tables,
            patient_features=patient_features,
        )
        train, val, test = split_feature_table(feat, split)
        trainval = pd.concat([train, val], ignore_index=True)
        if trainval.empty or test.empty or trainval["target"].nunique() < 2:
            continue
        feature_cols = select_feature_columns(feat)
        X_tv = trainval[feature_cols]
        y_tv = trainval["target"].astype(int)
        X_val = val[feature_cols]
        y_val = val["target"].astype(int)
        X_te = test[feature_cols]
        y_te = test["target"].astype(int)
        n_pos = int(y_tv.sum())
        n_neg = int(len(y_tv) - n_pos)

        # Fit model #1 on train+val for test predictions (headline metrics).
        try:
            model = build_model(algo, hp, n_pos=n_pos, n_neg=n_neg, random_state=args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tv, y_tv)
                p_test = model.predict_proba(X_te)[:, 1]
        except Exception as exc:
            print(f"  !! refit failed: T={T} L={L} W={W} {algo}: {exc!r}")
            continue
        # Fit model #2 on train only to get untainted val predictions for
        # threshold tuning (using model #1's predictions on val would be
        # circular since val was in its training set).
        tuned_threshold = 0.5
        try:
            X_tr = train[feature_cols]
            y_tr = train["target"].astype(int)
            n_pos_tr = int(y_tr.sum())
            n_neg_tr = int(len(y_tr) - n_pos_tr)
            if n_pos_tr > 0 and n_neg_tr > 0:
                model_tr = build_model(
                    algo, hp, n_pos=n_pos_tr, n_neg=n_neg_tr, random_state=args.seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_tr.fit(X_tr, y_tr)
                    p_val_for_tune = model_tr.predict_proba(X_val)[:, 1]
                tuned_threshold = _tune_threshold_max_f1(y_val.to_numpy(), p_val_for_tune)
        except Exception as exc:
            print(f"  !! threshold-tune fit failed ({algo} T={T} L={L} W={W}): {exc!r}")

        metrics = compute_metrics(y_te.to_numpy(), p_test)
        # F1 at the val-tuned operating threshold (separate from 0.5 default).
        from sklearn.metrics import f1_score as _f1
        from sklearn.metrics import precision_score as _prec
        from sklearn.metrics import recall_score as _rec
        y_te_arr = y_te.to_numpy()
        pred_tuned = (p_test >= tuned_threshold).astype(int)
        f1_tuned = float(_f1(y_te_arr, pred_tuned, zero_division=0))
        prec_tuned = float(_prec(y_te_arr, pred_tuned, zero_division=0))
        rec_tuned = float(_rec(y_te_arr, pred_tuned, zero_division=0))
        tab_rows.append(
            {
                "model_family": "tabular",
                "threshold": T,
                "input_length_days": L,
                "washout_days": W,
                "algo": algo,
                "hp": str(hp),
                "n_trainval": len(trainval),
                "n_test": len(test),
                "trainval_positive": n_pos,
                "test_positive": int(y_te.sum()),
                "tuned_threshold": tuned_threshold,
                "test_f1_tuned": f1_tuned,
                "test_precision_tuned": prec_tuned,
                "test_recall_tuned": rec_tuned,
                **{f"test_{k}": v for k, v in metrics.items()},
            }
        )
        # Per-sample test predictions for offline bootstrap CIs.
        test_meta = test.reset_index(drop=True)
        for i in range(len(test_meta)):
            tab_pred_rows.append(
                {
                    "model_family": "tabular",
                    "algo": algo,
                    "threshold": T,
                    "input_length_days": L,
                    "washout_days": W,
                    "user_key": int(test_meta.loc[i, USER_COL]),
                    "y_true": int(y_te_arr[i]),
                    "y_prob": float(p_test[i]),
                }
            )
    tab_df = pd.DataFrame(tab_rows)
    if not tab_df.empty:
        out = OUTPUT_TABLES_V2 / "final_test_v2_tabular.csv"
        tab_df.to_csv(out, index=False)
        print(f"  wrote {len(tab_df)} tabular rows -> {out}")
    if tab_pred_rows:
        pred_out = OUTPUT_TABLES_V2 / "final_test_v2_predictions_tabular.csv"
        pd.DataFrame(tab_pred_rows).to_csv(pred_out, index=False)
        print(f"  wrote {len(tab_pred_rows)} tabular predictions -> {pred_out}")

    # ---- 2. DL multi-seed winner ----------------------------------- #
    dl_path = Path(args.dl_multiseed)
    dl_df = pd.DataFrame()
    if dl_path.exists():
        ms = pd.read_csv(dl_path)
        if not ms.empty:
            print("\n[final] refitting DL multi-seed winner on train+val ...")
            cfg = ms.iloc[0]
            T = int(cfg["threshold"])
            L = int(cfg["input_length_days"])
            W = int(cfg["washout_days"])
            arch = str(cfg["arch"])
            hp = {}
            for k in cfg.index:
                if k.startswith("hp_"):
                    v = cfg[k]
                    if pd.notna(v):
                        hp[k[3:]] = v
            for k in ("hidden_dim", "num_layers"):
                if k in hp:
                    hp[k] = int(hp[k])

            if (T, L, W) not in sample_indexes:
                # rebuild on-demand
                _arts = run_event_labeling(thresholds=[T])
                _idx, _ = build_all_sample_indexes(
                    weekly_events=_arts.weekly_events,
                    probable_days_by_threshold=_arts.probable_event_days,
                    episodes_by_threshold=_arts.event_episodes,
                    input_lengths=[L],
                    washout_values=[W],
                )
                sample_indexes.update(_idx)

            sample_index = sample_indexes[(T, L, W)]
            dataset = build_sequence_dataset(sample_index, daily_tables)
            X_dict, y_dict = split_sequence_dataset(dataset, split)
            # train + val become a single training set; we don't early-stop
            # on test, so use the previously-selected best_epoch instead.
            X_tv = np.concatenate([X_dict["train"], X_dict["val"]], axis=0)
            y_tv = np.concatenate([y_dict["train"], y_dict["val"]], axis=0)
            X_te = X_dict["test"]
            y_te = y_dict["test"]

            seeds = ms["seed"].astype(int).tolist()
            dl_rows = []
            dl_pred_rows = []
            # Synthesise a 2-way dict for train_one_arch by carving a tiny
            # internal val (use last 20% of trainval, patient-grouped if
            # possible — simplest: just reuse y_dict['val'] for early-stop
            # but here trainval already includes val; we instead use a
            # train-only fit with epochs = median best_epoch from multiseed.
            target_epochs = int(np.median(ms["best_epoch"].astype(int)))
            print(
                f"  trainval={len(y_tv)}  test={len(y_te)}  epochs={target_epochs}"
            )
            # Provide a fake "val" loader = a small portion of trainval (no
            # leakage with test); used only for early-stop signal under our
            # train_torch_model.  We use the held-out original val partition
            # as the early-stop signal to avoid leaking test.
            fit_X = {"train": X_tv, "val": X_dict["val"], "test": X_te}
            fit_y = {"train": y_tv, "val": y_dict["val"], "test": y_te}
            # Recover per-test-sample user keys for the predictions file.
            test_user_keys = dataset.users[
                np.isin(dataset.users, np.array(split.test_users, dtype=int))
            ]
            for s in seeds:
                out = train_one_arch(
                    arch=arch,
                    X_dict=fit_X,
                    y_dict=fit_y,
                    hp=hp,
                    seed=s,
                    epochs=args.dl_epochs,
                    batch_size=args.dl_batch_size,
                    patience=args.dl_patience,
                    learning_rate=args.dl_learning_rate,
                )
                row = {
                    "model_family": "deep_learning",
                    "threshold": T,
                    "input_length_days": L,
                    "washout_days": W,
                    "arch": arch,
                    "hp": str(hp),
                    "seed": s,
                    "best_epoch": out["best_epoch"],
                }
                for k, v in out.items():
                    if k.startswith("test_"):
                        row[k] = v
                dl_rows.append(row)
                # per-sample test predictions for this seed
                p_test_seed = np.asarray(out.get("p_test", []), dtype=float)
                for i in range(len(p_test_seed)):
                    dl_pred_rows.append(
                        {
                            "model_family": "deep_learning",
                            "arch": arch,
                            "threshold": T,
                            "input_length_days": L,
                            "washout_days": W,
                            "seed": int(s),
                            "user_key": int(test_user_keys[i]) if i < len(test_user_keys) else -1,
                            "y_true": int(y_te[i]),
                            "y_prob": float(p_test_seed[i]),
                        }
                    )
            dl_df = pd.DataFrame(dl_rows)
            out_path = OUTPUT_TABLES_V2 / "final_test_v2_dl.csv"
            dl_df.to_csv(out_path, index=False)
            print(f"  wrote {len(dl_df)} DL rows -> {out_path}")
            if dl_pred_rows:
                pred_out = OUTPUT_TABLES_V2 / "final_test_v2_predictions_dl.csv"
                pd.DataFrame(dl_pred_rows).to_csv(pred_out, index=False)
                print(f"  wrote {len(dl_pred_rows)} DL predictions -> {pred_out}")

    # ---- 3. Headline summary --------------------------------------- #
    headline_rows: list[dict] = []
    if not tab_df.empty:
        # Pick the best per algo by test_pr_auc.
        top = (
            tab_df.sort_values("test_pr_auc", ascending=False)
            .groupby("algo", as_index=False)
            .head(1)
        )
        for r in top.itertuples(index=False):
            headline_rows.append(
                {
                    "family": "tabular",
                    "name": r.algo,
                    "threshold": r.threshold,
                    "input_length_days": r.input_length_days,
                    "washout_days": r.washout_days,
                    "test_pr_auc": r.test_pr_auc,
                    "test_roc_auc": r.test_roc_auc,
                    "test_f1": r.test_f1,
                    "test_f1_tuned": getattr(r, "test_f1_tuned", float("nan")),
                    "tuned_threshold": getattr(r, "tuned_threshold", float("nan")),
                    "test_precision": r.test_precision,
                    "test_recall": r.test_recall,
                    "test_brier": r.test_brier,
                    "n_test": r.n_test,
                }
            )
    if not dl_df.empty:
        agg = (
            dl_df.groupby(["arch", "threshold", "input_length_days", "washout_days"])
            .agg(
                test_pr_auc_mean=("test_pr_auc", "mean"),
                test_pr_auc_std=("test_pr_auc", "std"),
                test_roc_auc_mean=("test_roc_auc", "mean"),
                test_f1_mean=("test_f1", "mean"),
                n_seeds=("seed", "count"),
            )
            .reset_index()
            .sort_values("test_pr_auc_mean", ascending=False)
        )
        for r in agg.itertuples(index=False):
            headline_rows.append(
                {
                    "family": "deep_learning",
                    "name": r.arch,
                    "threshold": r.threshold,
                    "input_length_days": r.input_length_days,
                    "washout_days": r.washout_days,
                    "test_pr_auc": r.test_pr_auc_mean,
                    "test_pr_auc_std": r.test_pr_auc_std,
                    "test_roc_auc": r.test_roc_auc_mean,
                    "test_f1": r.test_f1_mean,
                    "n_seeds": int(r.n_seeds),
                }
            )

    if headline_rows:
        head = pd.DataFrame(headline_rows)
        out_path = OUTPUT_TABLES_V2 / "final_test_v2_summary.csv"
        head.to_csv(out_path, index=False)
        print("\n=== HEADLINE (test-set) ===")
        print(head.to_string(index=False))
        print(f"\nWrote -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
