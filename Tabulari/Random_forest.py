# Trees.py  -- baseline ad alberi sugli STESSI split e con le STESSE metriche di MLP.py / RegLog.py
import argparse
import os
import joblib

from dataset import set_seed, load_splits, make_loaders
from metrics_report import predict_proba, find_threshold, evaluate_and_save


def build_model(kind, pos_weight, seed, n_jobs):
    use_w = pos_weight is not None

    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,          # evita foglie quasi pure: con dati rumorosi migliora molto
            max_features="sqrt",
            class_weight="balanced_subsample" if use_w else None,
            n_jobs=n_jobs,
            random_state=seed,
        )

    if kind == "hgb":
        # Gradient boosting di scikit-learn: nessuna installazione extra
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=30,
            class_weight="balanced" if use_w else None,
            random_state=seed,
        )

    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise SystemExit("LightGBM non installato. Dentro il container: pip install --user lightgbm "
                             "(oppure usa --model hgb).")
        return LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight if use_w else 1.0,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )

    raise ValueError(kind)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/heart_disease_health_indicators.csv")
    ap.add_argument("--model", choices=["rf", "lgbm", "hgb"], default="lgbm")
    ap.add_argument("--no_pos_weight", action="store_true")
    ap.add_argument("--min_recall", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"CPU usate: {n_jobs}")

    splits = load_splits(args.csv, seed=args.seed)
    _, val_loader, test_loader = make_loaders(splits, batch_size=4096)

    pw = None if args.no_pos_weight else splits["pos_weight"]
    model = build_model(args.model, pw, args.seed, n_jobs)
    model_name = f"{args.model}{'_nopw' if args.no_pos_weight else ''}"

    print(f"Addestramento di {model_name}...")
    if args.model == "lgbm":
        import lightgbm as lgb
        model.fit(
            splits["X_train"], splits["y_train"],
            eval_set=[(splits["X_val"], splits["y_val"])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        print(f"Alberi usati (early stopping): {model.best_iteration_}")
    else:
        model.fit(splits["X_train"], splits["y_train"])
    print("Addestramento completato!")

    # Soglia scelta sul validation, test usato una sola volta
    y_val, p_val = predict_proba(model, val_loader)
    thr = find_threshold(y_val, p_val, min_recall=args.min_recall)
    print(f"Soglia scelta sul validation: {thr:.3f}")

    evaluate_and_save(model, test_loader, model_name=model_name, threshold=thr)

    os.makedirs("weight", exist_ok=True)
    joblib.dump(model, f"weight/{model_name}.joblib")