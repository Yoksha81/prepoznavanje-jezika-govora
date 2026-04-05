from pathlib import Path
import json
import cv2 as cv
import numpy as np
import pandas as pd


def osiguraj_direktorijum(putanja: Path) -> None:
    putanja.mkdir(parents=True, exist_ok=True)


def ucitaj_manifest(putanja):
    return pd.read_csv(putanja)


def nadji_excel_fajl(speaker_dir: Path) -> Path:
    excel_fajlovi = sorted(speaker_dir.glob("*.xlsx"))
    if not excel_fajlovi:
        raise FileNotFoundError(f"Nema excel fajla u: {speaker_dir}")
    return excel_fajlovi[0]


def bezbedan_string(vrednost) -> str:
    if pd.isna(vrednost):
        return ""
    return str(vrednost).strip()


def normalizuj_jezik(jezik) -> str:
    s = str(jezik).strip().lower()
    if s in {"ser", "sr", "serbian", "srb"}:
        return "ser"
    if s in {"eng", "en", "english"}:
        return "eng"
    return s


def u_binarni_flag(vrednost) -> int:
    s = str(vrednost).strip().lower()
    if s in {"no", "nan", "", "false", "0", "none"}:
        return 0
    return 1


def ucitaj_align_fajl(align_path):
    return pd.read_csv(
        align_path,
        sep=r"\s+",
        header=None,
        names=["start", "end", "token"],
        engine="python",
    )


def interval_govora_iz_align(align_path, align_scale):
    df = ucitaj_align_fajl(align_path)
    govor_df = df[df["token"] != "sil"].copy()
    if govor_df.empty:
        return None, None
    start_s = float(govor_df.iloc[0]["start"]) / align_scale
    end_s = float(govor_df.iloc[-1]["end"]) / align_scale
    return start_s, end_s


def ekvidistantne_tacke(start_s, end_s, n_frejmova, mode="centers"):
    if end_s <= start_s:
        return None
    if mode == "linspace":
        return np.linspace(start_s, end_s, n_frejmova)
    if mode == "centers":
        ivice = np.linspace(start_s, end_s, n_frejmova + 1)
        return (ivice[:-1] + ivice[1:]) / 2.0
    raise ValueError("mode mora biti 'linspace' ili 'centers'")


def vremena_u_frejmove(vremena_s, fps, frame_count):
    frejmovi = np.round(vremena_s * fps).astype(int)
    frejmovi = np.clip(frejmovi, 0, frame_count - 1)
    return frejmovi


def ucitaj_roi_fajl(roi_path):
    roi_df = pd.read_csv(roi_path)
    roi_df.columns = [c.strip().lower() for c in roi_df.columns]
    return roi_df


def nadji_roi_za_frejm(roi_df, frame_idx):
    roi_frame = int(frame_idx) + 1
    red = roi_df[roi_df["frame"] == roi_frame]
    if red.empty:
        return None
    red = red.iloc[0]
    return int(red["x"]), int(red["y"]), int(red["width"]), int(red["height"])


def otvori_video(video_path):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return cap, fps, frame_count


def procitaj_frejm(cap, frame_idx):
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def iseci_roi(frame, roi, padding=0):
    x, y, w, h = roi
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def resize_roi(crop, out_size, interpolation=cv.INTER_AREA):
    return cv.resize(crop, out_size, interpolation=interpolation)


def sacuvaj_sliku(putanja, image):
    return cv.imwrite(str(putanja), image)


def stringify_params(parametri):
    izlaz = {}
    for k, v in parametri.items():
        if isinstance(v, tuple):
            izlaz[k] = list(v)
        elif isinstance(v, np.generic):
            izlaz[k] = v.item()
        else:
            izlaz[k] = v
    return izlaz


def ucitaj_feature_tabelu(features_dir, prefix, kamera):
    meta_path = features_dir / f"{prefix}_{kamera}_metadata.csv"
    feat_path = features_dir / f"{prefix}_{kamera}_features.npy"
    meta_df = pd.read_csv(meta_path)
    X = np.load(feat_path)
    feat_cols = [f"{prefix}_f_{i}" for i in range(X.shape[1])]
    feat_df = pd.DataFrame(X, columns=feat_cols)
    return pd.concat([meta_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def spoji_hog_i_resnet(features_dir, hog_prefix, resnet_prefix, kamera):
    hog_df = ucitaj_feature_tabelu(features_dir, hog_prefix, kamera)
    resnet_df = ucitaj_feature_tabelu(features_dir, resnet_prefix, kamera)
    merge_cols = ["speaker", "sample_name", "language", "split", "camera", "n_crops"]
    return hog_df.merge(resnet_df, on=merge_cols, how="inner", suffixes=("_hogmeta", "_resmeta"))


def vrati_skupove_obelezja(df, hog_prefix, resnet_prefix):
    hog_cols = [c for c in df.columns if c.startswith(f"{hog_prefix}_f_")]
    resnet_cols = [c for c in df.columns if c.startswith(f"{resnet_prefix}_f_")]
    return {
        "hog_only": hog_cols,
        "resnet18_only": resnet_cols,
        "hog_plus_resnet18": hog_cols + resnet_cols,
    }


def ucitaj_json_parametre(params_json):
    return json.loads(params_json)


def castuj_parametre_za_pipeline(params_dict):
    casted = {}
    for k, v in params_dict.items():
        if k == "clf__hidden_layer_sizes" and isinstance(v, list):
            casted[k] = tuple(v)
        else:
            casted[k] = v
    return casted
