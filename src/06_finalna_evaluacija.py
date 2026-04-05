import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from konfiguracija import (
    FEATURES_DIR, FINAL_RESULTS_DIR, BEST_PARAMS_PATH, CV_RESULTS_PATH,
    HOG_PREFIX, RESNET_PREFIX
)
from funkcije import (
    osiguraj_direktorijum, spoji_hog_i_resnet, vrati_skupove_obelezja,
    ucitaj_json_parametre, castuj_parametre_za_pipeline
)

KAMERE = ["A", "L", "R"]
LABEL_MAP = {"ser": 0, "eng": 1}


def napravi_pipeline(model_name):
    if model_name == "svm":
        return Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=42)), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])
    if model_name == "rf":
        return Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=42)), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))])
    if model_name == "mlp":
        return Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=42)), ("clf", MLPClassifier(max_iter=600, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1, random_state=42))])
    raise ValueError(f"Nepoznat model: {model_name}")


def izracunaj_metrike(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def sacuvaj_matricu_konfuzije(y_true, y_pred, naslov, putanja):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ser", "eng"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Oranges", values_format="d")
    plt.title(naslov)
    plt.tight_layout()
    plt.savefig(putanja, dpi=150)
    plt.close(fig)


def strogi_izbor_po_kameri(cv_results_df, gap_weight=0.5, std_weight=0.1):
    df = cv_results_df.copy()
    df["gap"] = df["mean_train_score"] - df["mean_test_score"]
    df["selection_score"] = df["mean_test_score"] - gap_weight * df["gap"] - std_weight * df["std_test_score"]
    idx = df.groupby("camera")["selection_score"].idxmax()
    selected = df.loc[idx].copy().reset_index(drop=True)
    selected = selected.rename(columns={"mean_test_score": "best_cv_score", "params_json": "best_params_json"})
    return selected[["camera", "experiment", "model", "n_features", "best_cv_score", "best_params_json", "mean_train_score", "std_test_score", "gap", "selection_score"]]


def homogeneous_fusion(predictions_df):
    fusion_rows, fusion_predictions = [], []
    for exp_name in sorted(predictions_df["experiment"].unique()):
        for model_name in sorted(predictions_df["model"].unique()):
            df_sub = predictions_df[(predictions_df["experiment"] == exp_name) & (predictions_df["model"] == model_name)].copy()
            if df_sub.empty:
                continue
            fusion_df = df_sub.groupby(["speaker", "sample_name", "language", "split"], as_index=False).agg({"prob_eng": "mean", "y_true": "first"})
            fusion_df["y_pred"] = (fusion_df["prob_eng"] >= 0.5).astype(int)
            fusion_df["experiment"] = exp_name
            fusion_df["model"] = model_name
            fusion_df["fusion_name"] = f"{exp_name}__{model_name}"
            acc, prec, rec, f1 = izracunaj_metrike(fusion_df["y_true"], fusion_df["y_pred"])
            fusion_rows.append({"fusion_type": "homogeneous", "fusion_name": f"{exp_name}__{model_name}", "camera": "A+L+R", "experiment": exp_name, "model": model_name, "best_cv_score": np.nan, "test_accuracy": acc, "test_precision_macro": prec, "test_recall_macro": rec, "test_f1_macro": f1})
            fusion_predictions.append(fusion_df)
    return pd.DataFrame(fusion_rows), pd.concat(fusion_predictions, ignore_index=True) if fusion_predictions else pd.DataFrame()


def best_of_best_fusion(predictions_df, best_view_df):
    selected_parts, config_rows = [], []
    for _, row in best_view_df.iterrows():
        sub = predictions_df[(predictions_df["camera"] == row["camera"]) & (predictions_df["experiment"] == row["experiment"]) & (predictions_df["model"] == row["model"])].copy()
        if sub.empty:
            continue
        selected_parts.append(sub)
        config_rows.append(row.to_dict())
    if not selected_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_df = pd.concat(selected_parts, ignore_index=True)
    fusion_df = selected_df.groupby(["speaker", "sample_name", "language", "split"], as_index=False).agg({"prob_eng": "mean", "y_true": "first"})
    fusion_df["y_pred"] = (fusion_df["prob_eng"] >= 0.5).astype(int)
    acc, prec, rec, f1 = izracunaj_metrike(fusion_df["y_true"], fusion_df["y_pred"])
    metrics_df = pd.DataFrame([{"fusion_type": "best_of_best", "fusion_name": "best_per_view_strict", "camera": "A+L+R", "experiment": "mixed", "model": "mixed", "best_cv_score": float(best_view_df["best_cv_score"].mean()), "test_accuracy": acc, "test_precision_macro": prec, "test_recall_macro": rec, "test_f1_macro": f1}])
    return metrics_df, fusion_df, pd.DataFrame(config_rows)


def main():
    osiguraj_direktorijum(FINAL_RESULTS_DIR)
    best_params_df = pd.read_csv(BEST_PARAMS_PATH)
    cv_results_df = pd.read_csv(CV_RESULTS_PATH)
    all_metrics, all_predictions = [], []
    for kamera in KAMERE:
        df = spoji_hog_i_resnet(FEATURES_DIR, HOG_PREFIX, RESNET_PREFIX, kamera)
        df["y"] = df["language"].map(LABEL_MAP)
        feature_sets = vrati_skupove_obelezja(df, HOG_PREFIX, RESNET_PREFIX)
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        y_train, y_test = train_df["y"], test_df["y"]
        for exp_name, feature_cols in feature_sets.items():
            X_train, X_test = train_df[feature_cols], test_df[feature_cols]
            for model_name in ["svm", "rf", "mlp"]:
                red = best_params_df[(best_params_df["camera"] == kamera) & (best_params_df["experiment"] == exp_name) & (best_params_df["model"] == model_name)]
                if red.empty:
                    continue
                params_dict = castuj_parametre_za_pipeline(ucitaj_json_parametre(red.iloc[0]["best_params_json"]))
                model = napravi_pipeline(model_name)
                model.set_params(**params_dict)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
                acc, prec, rec, f1 = izracunaj_metrike(y_test, y_pred)
                all_metrics.append({"camera": kamera, "experiment": exp_name, "model": model_name, "n_features": len(feature_cols), "best_cv_score": float(red.iloc[0]["best_cv_score"]), "best_params_json": red.iloc[0]["best_params_json"], "test_accuracy": acc, "test_precision_macro": prec, "test_recall_macro": rec, "test_f1_macro": f1})
                pred_df = test_df[["speaker", "sample_name", "language", "split"]].copy()
                pred_df["camera"] = kamera
                pred_df["experiment"] = exp_name
                pred_df["model"] = model_name
                pred_df["y_true"] = y_test.values
                pred_df["y_pred"] = y_pred
                pred_df["prob_eng"] = y_prob
                all_predictions.append(pred_df)
                sacuvaj_matricu_konfuzije(y_test, y_pred, f"{kamera} | {exp_name} | {model_name}", FINAL_RESULTS_DIR / f"cm_{kamera}_{exp_name}_{model_name}.png")
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(FINAL_RESULTS_DIR / "metrics_per_camera.csv", index=False)
    predictions_df.to_csv(FINAL_RESULTS_DIR / "predictions_per_camera.csv", index=False)
    fusion_metrics_h, fusion_predictions_h = homogeneous_fusion(predictions_df)
    fusion_metrics_h.to_csv(FINAL_RESULTS_DIR / "metrics_fusion_homogeneous.csv", index=False)
    if not fusion_predictions_h.empty:
        fusion_predictions_h.to_csv(FINAL_RESULTS_DIR / "predictions_fusion_homogeneous_all.csv", index=False)
        for _, red in fusion_metrics_h.iterrows():
            sub = fusion_predictions_h[(fusion_predictions_h["experiment"] == red["experiment"]) & (fusion_predictions_h["model"] == red["model"])]
            sacuvaj_matricu_konfuzije(sub["y_true"], sub["y_pred"], f"fusion | {red['experiment']} | {red['model']}", FINAL_RESULTS_DIR / f"cm_fusion_homogeneous_{red['experiment']}_{red['model']}.png")
    best_view_df = strogi_izbor_po_kameri(cv_results_df)
    best_view_df.to_csv(FINAL_RESULTS_DIR / "best_model_per_camera_from_cv_strict.csv", index=False)
    fusion_metrics_best, fusion_predictions_best, fusion_config_best = best_of_best_fusion(predictions_df, best_view_df)
    if not fusion_metrics_best.empty:
        fusion_metrics_best.to_csv(FINAL_RESULTS_DIR / "metrics_fusion_best_of_best_strict.csv", index=False)
        fusion_predictions_best.to_csv(FINAL_RESULTS_DIR / "fusion_predictions_best_of_best_strict.csv", index=False)
        fusion_config_best.to_csv(FINAL_RESULTS_DIR / "fusion_best_of_best_strict_config.csv", index=False)
        sacuvaj_matricu_konfuzije(fusion_predictions_best["y_true"], fusion_predictions_best["y_pred"], "fusion | best_per_view_strict", FINAL_RESULTS_DIR / "cm_fusion_best_of_best_strict.png")
    pd.concat([metrics_df, fusion_metrics_h, fusion_metrics_best], ignore_index=True, sort=False).to_csv(FINAL_RESULTS_DIR / "metrics_all.csv", index=False)
    print("Finalna evaluacija je zavrsena.")


if __name__ == "__main__":
    main()
