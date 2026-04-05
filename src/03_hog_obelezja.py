import cv2 as cv
import numpy as np
import pandas as pd
from skimage.feature import hog

from konfiguracija import (
    CROP_MANIFEST,
    FEATURES_DIR,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    USE_DELTA_FEATURES,
    ROI_OUT_SIZE_BY_CAMERA,
)
from funkcije import osiguraj_direktorijum

KAMERE = ["A", "L", "R"]


def ucitaj_crop_manifest():
    return pd.read_csv(CROP_MANIFEST)


def ucitaj_i_resize_za_hog(crop_path, camera):
    img = cv.imread(str(crop_path))
    if img is None:
        return None
    out_size = ROI_OUT_SIZE_BY_CAMERA[camera]
    img_resized = cv.resize(img, out_size, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    return gray


def izracunaj_hog_za_sliku(crop_path, camera):
    gray = ucitaj_i_resize_za_hog(crop_path, camera)
    if gray is None:
        return None
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def agregiraj_static(hog_seq):
    return np.concatenate([hog_seq.mean(axis=0), hog_seq.std(axis=0)])


def izracunaj_delta(hog_seq):
    if hog_seq.shape[0] < 2:
        return None
    return np.diff(hog_seq, axis=0)


def agregiraj_delta(delta_seq):
    return np.concatenate([np.abs(delta_seq).mean(axis=0), delta_seq.std(axis=0)])


def napravi_video_feature(group_df):
    group_df = group_df.sort_values("crop_order").copy()
    if len(group_df) != 40:
        return None
    camera = group_df["camera"].iloc[0]
    hog_lista = []
    for _, row in group_df.iterrows():
        feat = izracunaj_hog_za_sliku(row["crop_path"], camera)
        if feat is None:
            return None
        hog_lista.append(feat)
    hog_seq = np.vstack(hog_lista)
    static_feat = agregiraj_static(hog_seq)
    if USE_DELTA_FEATURES:
        delta_seq = izracunaj_delta(hog_seq)
        final_feat = static_feat if delta_seq is None else np.concatenate([static_feat, agregiraj_delta(delta_seq)])
    else:
        final_feat = static_feat
    return final_feat, hog_seq.shape[0], hog_seq.shape[1]


def sacuvaj_dataset_za_kameru(meta_df, X, kamera):
    osiguraj_direktorijum(FEATURES_DIR)
    suffix = "hog_static_delta" if USE_DELTA_FEATURES else "hog_static"
    meta_path = FEATURES_DIR / f"{suffix}_{kamera}_metadata.csv"
    npy_path = FEATURES_DIR / f"{suffix}_{kamera}_features.npy"
    full_csv_path = FEATURES_DIR / f"{suffix}_{kamera}_full.csv"
    meta_df.to_csv(meta_path, index=False)
    np.save(npy_path, X)
    feat_cols = [f"hog_{i}" for i in range(X.shape[1])]
    pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(X, columns=feat_cols)], axis=1).to_csv(full_csv_path, index=False)
    print(f"KAMERA {kamera} | oblik feature matrice: {X.shape}")


def obradi_kameru(crop_df, kamera):
    kamera_df = crop_df[crop_df["camera"] == kamera].copy()
    if kamera_df.empty:
        return
    grouped = kamera_df.groupby(["speaker", "sample_name", "language", "split", "camera"])
    metadata_rows, feature_rows = [], []
    for key, group_df in grouped:
        speaker, sample_name, language, split, camera = key
        rezultat = napravi_video_feature(group_df)
        if rezultat is None:
            continue
        final_feat, n_crops, frame_hog_dim = rezultat
        metadata_rows.append({
            "speaker": speaker,
            "sample_name": sample_name,
            "language": language,
            "split": split,
            "camera": camera,
            "n_crops": n_crops,
            "frame_hog_dim": frame_hog_dim,
            "video_feature_dim": len(final_feat),
        })
        feature_rows.append(final_feat)
    if feature_rows:
        sacuvaj_dataset_za_kameru(pd.DataFrame(metadata_rows), np.vstack(feature_rows), kamera)


def main():
    crop_df = pd.read_csv(CROP_MANIFEST)
    for kamera in KAMERE:
        obradi_kameru(crop_df, kamera)


if __name__ == "__main__":
    main()
