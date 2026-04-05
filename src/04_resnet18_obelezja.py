import warnings
warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

from konfiguracija import CROP_MANIFEST, FEATURES_DIR, RESNET_INPUT_SIZE, RESNET_USE_STD_AGG
from funkcije import osiguraj_direktorijum

KAMERE = ["A", "L", "R"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 40


def ucitaj_crop_manifest():
    return pd.read_csv(CROP_MANIFEST)


def napravi_model_i_transform():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    return model, transform


def ucitaj_i_transformisi_sliku(crop_path, transform):
    img_bgr = cv.imread(str(crop_path))
    if img_bgr is None:
        return None
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return transform(img_rgb)


@torch.no_grad()
def izracunaj_resnet_embeddinge_za_cropove(crop_paths, model, transform, batch_size=BATCH_SIZE):
    tensori = []
    for path in crop_paths:
        t = ucitaj_i_transformisi_sliku(path, transform)
        if t is None:
            return None
        tensori.append(t)
    if not tensori:
        return None
    X = torch.stack(tensori, dim=0)
    embeddings = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size].to(DEVICE)
        emb = model(xb)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def agregiraj_video_embedding(frame_embs):
    mean_feat = frame_embs.mean(axis=0)
    if RESNET_USE_STD_AGG:
        return np.concatenate([mean_feat, frame_embs.std(axis=0)])
    return mean_feat


def obradi_kameru(crop_df, kamera, model, transform):
    kamera_df = crop_df[crop_df["camera"] == kamera].copy()
    if kamera_df.empty:
        return
    grouped = kamera_df.groupby(["speaker", "sample_name", "language", "split", "camera"])
    metadata_rows, feature_rows = [], []
    for key, group_df in grouped:
        speaker, sample_name, language, split, camera = key
        group_df = group_df.sort_values("crop_order").copy()
        if len(group_df) != 40:
            continue
        crop_paths = group_df["crop_path"].tolist()
        frame_embs = izracunaj_resnet_embeddinge_za_cropove(crop_paths, model, transform)
        if frame_embs is None:
            continue
        video_feat = agregiraj_video_embedding(frame_embs)
        metadata_rows.append({
            "speaker": speaker,
            "sample_name": sample_name,
            "language": language,
            "split": split,
            "camera": camera,
            "n_crops": len(group_df),
            "frame_emb_dim": frame_embs.shape[1],
            "video_feature_dim": len(video_feat),
        })
        feature_rows.append(video_feat)
    if not feature_rows:
        return
    meta_df = pd.DataFrame(metadata_rows)
    X = np.vstack(feature_rows)
    osiguraj_direktorijum(FEATURES_DIR)
    suffix = "resnet18_meanstd" if RESNET_USE_STD_AGG else "resnet18_mean"
    meta_path = FEATURES_DIR / f"{suffix}_{kamera}_metadata.csv"
    npy_path = FEATURES_DIR / f"{suffix}_{kamera}_features.npy"
    full_csv_path = FEATURES_DIR / f"{suffix}_{kamera}_full.csv"
    meta_df.to_csv(meta_path, index=False)
    np.save(npy_path, X)
    feat_cols = [f"resnet18_{i}" for i in range(X.shape[1])]
    pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(X, columns=feat_cols)], axis=1).to_csv(full_csv_path, index=False)
    print(f"KAMERA {kamera} | oblik feature matrice: {X.shape}")


def main():
    crop_df = ucitaj_crop_manifest()
    model, transform = napravi_model_i_transform()
    for kamera in KAMERE:
        obradi_kameru(crop_df, kamera, model, transform)


if __name__ == "__main__":
    main()
