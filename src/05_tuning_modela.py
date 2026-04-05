import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from konfiguracija import (
    FEATURES_DIR, TUNING_DIR, CV_RESULTS_PATH, BEST_PARAMS_PATH,
    BEST_PARAMS_SUMMARY_PATH, HOG_PREFIX, RESNET_PREFIX
)
from funkcije import (
    osiguraj_direktorijum, spoji_hog_i_resnet, vrati_skupove_obelezja,
    stringify_params
)

KAMERE = ["A", "L", "R"]
LABEL_MAP = {"ser": 0, "eng": 1}


def napravi_svm_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])


def napravi_rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
    ])


def napravi_mlp_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", MLPClassifier(max_iter=600, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1, random_state=42)),
    ])


def vrati_pune_gridove():
    return {
            "svm": {
            "pipe": napravi_svm_pipeline(),
            "params": {
                "pca__n_components": [0.9, 0.95, 128, 256, 384],
                "clf__C": [0.25, 0.5, 1, 4],
                "clf__gamma": ["scale", 0.001],
            },
        },

        "rf": {
            "pipe": napravi_rf_pipeline(),
            "params": {
                "pca__n_components": [0.95, 128, 256],
                "clf__max_depth": [None, 20, 30],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        },

        "mlp": {
            "pipe": napravi_mlp_pipeline(),
            "params": {
                "pca__n_components": [0.95, 128, 256],
                "clf__hidden_layer_sizes": [(128,), (256,), (256, 128)],
                "clf__alpha": [0.0001, 0.001, 0.05],
                "clf__learning_rate_init": [0.001],
                "clf__batch_size": [32],
            },
        },
    }


def vrati_refinement_gridove():
    return {
        ("R", "hog_only", "rf"): {
            "pipe": napravi_rf_pipeline(),
            "params": {
                "pca__n_components": [64, 256],
                "clf__n_estimators": [400],
                "clf__max_depth": [4, 8],
                "clf__min_samples_split": [20, 30],
                "clf__min_samples_leaf": [4, 8],
                "clf__max_features": ["sqrt", 0.3],
                "clf__class_weight": ["balanced", "balanced_subsample"],
            },
        },
    }


def pripremi_df_za_kameru(kamera):
    df = spoji_hog_i_resnet(FEATURES_DIR, HOG_PREFIX, RESNET_PREFIX, kamera)
    df["y"] = df["language"].map(LABEL_MAP)
    return df


def dodaj_cv_redove(cv_results, kamera, eksperiment, model, broj_obelezja):
    keep_cols = ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "rank_test_score", "params"]
    cv_results = pd.DataFrame(cv_results)[keep_cols].copy()
    cv_results["camera"] = kamera
    cv_results["experiment"] = eksperiment
    cv_results["model"] = model
    cv_results["n_features"] = broj_obelezja
    cv_results["params_json"] = cv_results["params"].apply(lambda x: json.dumps(stringify_params(x), sort_keys=True))
    return cv_results.drop(columns=["params"])


def rekonstruisi_best_tabele(svi_cv_df):
    idx = svi_cv_df.groupby(["camera", "experiment", "model", "n_features"])["mean_test_score"].idxmax()
    best_df = svi_cv_df.loc[idx].copy().reset_index(drop=True)
    best_df = best_df.rename(columns={"mean_test_score": "best_cv_score", "params_json": "best_params_json"})
    best_df = best_df[["camera", "experiment", "model", "n_features", "best_cv_score", "best_params_json"]]
    summary_df = best_df.sort_values(["camera", "experiment", "best_cv_score"], ascending=[True, True, False]).reset_index(drop=True)
    return best_df, summary_df


def pokreni_puni_grid():
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    modeli = vrati_pune_gridove()
    svi_redovi = []
    for kamera in KAMERE:
        df = pripremi_df_za_kameru(kamera)
        feature_sets = vrati_skupove_obelezja(df, HOG_PREFIX, RESNET_PREFIX)
        train_df = df[df["split"] == "train"].copy()
        y_train = train_df["y"]
        groups_train = train_df["speaker"]
        for exp_name, feature_cols in feature_sets.items():
            X_train = train_df[feature_cols]
            for model_name, obj in modeli.items():
                grid = GridSearchCV(obj["pipe"], obj["params"], scoring="accuracy", cv=cv, n_jobs=-1, refit=True, return_train_score=True)
                grid.fit(X_train, y_train, groups=groups_train)
                svi_redovi.append(dodaj_cv_redove(grid.cv_results_, kamera, exp_name, model_name, len(feature_cols)))
    combined = pd.concat(svi_redovi, ignore_index=True)
    best_df, summary_df = rekonstruisi_best_tabele(combined)
    combined.to_csv(CV_RESULTS_PATH, index=False)
    best_df.to_csv(BEST_PARAMS_PATH, index=False)
    summary_df.to_csv(BEST_PARAMS_SUMMARY_PATH, index=False)


def pokreni_refinement():
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    configs = vrati_refinement_gridove()
    novi_redovi = []
    kamera_cache = {}
    for kamera in sorted(set(k[0] for k in configs.keys())):
        kamera_cache[kamera] = pripremi_df_za_kameru(kamera)
    for (kamera, exp_name, model_name), obj in configs.items():
        df = kamera_cache[kamera]
        feature_sets = vrati_skupove_obelezja(df, HOG_PREFIX, RESNET_PREFIX)
        train_df = df[df["split"] == "train"].copy()
        X_train = train_df[feature_sets[exp_name]]
        y_train = train_df["y"]
        groups_train = train_df["speaker"]
        grid = GridSearchCV(obj["pipe"], obj["params"], scoring="accuracy", cv=cv, n_jobs=-1, refit=True, return_train_score=True)
        grid.fit(X_train, y_train, groups=groups_train)
        novi_redovi.append(dodaj_cv_redove(grid.cv_results_, kamera, exp_name, model_name, len(feature_sets[exp_name])))
    novi_df = pd.concat(novi_redovi, ignore_index=True) if novi_redovi else pd.DataFrame()
    if CV_RESULTS_PATH.exists():
        postojeci = pd.read_csv(CV_RESULTS_PATH)
        combined = pd.concat([postojeci, novi_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["camera", "experiment", "model", "n_features", "params_json"], keep="last").reset_index(drop=True)
    else:
        combined = novi_df
    best_df, summary_df = rekonstruisi_best_tabele(combined)
    combined.to_csv(CV_RESULTS_PATH, index=False)
    best_df.to_csv(BEST_PARAMS_PATH, index=False)
    summary_df.to_csv(BEST_PARAMS_SUMMARY_PATH, index=False)


def main(rezim="puni"):
    osiguraj_direktorijum(TUNING_DIR)
    if rezim == "puni":
        pokreni_puni_grid()
    elif rezim == "refinement":
        pokreni_refinement()
    else:
        raise ValueError("Režim mora biti 'puni' ili 'refinement'")


if __name__ == "__main__":
    print("Pokrece se puni grid")
    main(rezim="puni")
    print("Pokrece se refinement grid")
    main(rezim="refinement")
    print("Tuning zavrsen")
