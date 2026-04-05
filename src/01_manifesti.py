from pathlib import Path
import pandas as pd

from konfiguracija import (
    RAW_DATA_DIR,
    MANIFESTS_DIR,
    TEST_SPEAKERS,
    COL_NAME,
    COL_LANGUAGE,
    COL_VIDEO_A,
    COL_VIDEO_L,
    COL_VIDEO_R,
    COL_COMMON,
    COL_TRANSCRIPT,
    VIDEO_EXT,
    ROI_EXT,
    ALIGN_EXT,
)
from funkcije import (
    osiguraj_direktorijum,
    nadji_excel_fajl,
    bezbedan_string,
    normalizuj_jezik,
    u_binarni_flag,
)


def napravi_putanje_uzorka(speaker_dir: Path, jezik: str, sample_name: str):
    lang_dir = "ser" if jezik == "ser" else "eng"
    putanja = speaker_dir / lang_dir

    video_a = putanja / "video_a_masked" / "video" / f"{sample_name}{VIDEO_EXT}"
    video_l = putanja / "video_l_masked" / "video" / f"{sample_name}{VIDEO_EXT}"
    video_r = putanja / "video_r_masked" / "video" / f"{sample_name}{VIDEO_EXT}"

    roi_a = putanja / "video_a_masked" / "roi" / f"{sample_name}{ROI_EXT}"
    roi_l = putanja / "video_l_masked" / "roi" / f"{sample_name}{ROI_EXT}"
    roi_r = putanja / "video_r_masked" / "roi" / f"{sample_name}{ROI_EXT}"

    align_path = speaker_dir / "alignment" / f"{sample_name}{ALIGN_EXT}"
    return video_a, video_l, video_r, roi_a, roi_l, roi_r, align_path


def napravi_pocetni_manifest():
    osiguraj_direktorijum(MANIFESTS_DIR)
    svi_redovi = []
    speaker_dirs = sorted([d for d in RAW_DATA_DIR.glob("spk*") if d.is_dir()])
    if not speaker_dirs:
        raise FileNotFoundError(f"Nema speaker foldera u {RAW_DATA_DIR}")

    for speaker_dir in speaker_dirs:
        speaker = speaker_dir.name
        split = "test" if speaker in TEST_SPEAKERS else "train"
        excel_path = nadji_excel_fajl(speaker_dir)
        df = pd.read_excel(excel_path)

        for _, row in df.iterrows():
            sample_name = bezbedan_string(row[COL_NAME])
            language = normalizuj_jezik(row[COL_LANGUAGE])
            common = row[COL_COMMON]
            transcript = bezbedan_string(row[COL_TRANSCRIPT]) if COL_TRANSCRIPT in df.columns else ""
            has_a = u_binarni_flag(row[COL_VIDEO_A])
            has_l = u_binarni_flag(row[COL_VIDEO_L])
            has_r = u_binarni_flag(row[COL_VIDEO_R])

            video_a, video_l, video_r, roi_a, roi_l, roi_r, align_path = napravi_putanje_uzorka(
                speaker_dir, language, sample_name
            )

            svi_redovi.append({
                "speaker": speaker,
                "sample_name": sample_name,
                "language": language,
                "common": common,
                "transcript": transcript,
                "split": split,
                "has_A": has_a,
                "has_L": has_l,
                "has_R": has_r,
                "video_a_path": str(video_a) if has_a else pd.NA,
                "video_l_path": str(video_l) if has_l else pd.NA,
                "video_r_path": str(video_r) if has_r else pd.NA,
                "roi_A_path": str(roi_a) if has_a else pd.NA,
                "roi_L_path": str(roi_l) if has_l else pd.NA,
                "roi_R_path": str(roi_r) if has_r else pd.NA,
                "align_path": str(align_path),
                "excel_path": str(excel_path),
                "video_A_exists": int(video_a.exists()) if has_a else 0,
                "video_L_exists": int(video_l.exists()) if has_l else 0,
                "video_R_exists": int(video_r.exists()) if has_r else 0,
                "roi_A_exists": int(roi_a.exists()) if has_a else 0,
                "roi_L_exists": int(roi_l.exists()) if has_l else 0,
                "roi_R_exists": int(roi_r.exists()) if has_r else 0,
                "align_exists": int(align_path.exists()),
            })

    manifest_df = pd.DataFrame(svi_redovi)
    manifest_df.to_csv(MANIFESTS_DIR / "manifest_all.csv", index=False)
    manifest_df[manifest_df["split"] == "train"].to_csv(MANIFESTS_DIR / "manifest_train.csv", index=False)
    manifest_df[manifest_df["split"] == "test"].to_csv(MANIFESTS_DIR / "manifest_test.csv", index=False)
    return manifest_df


def izvestaj_za_kameru(df, cam):
    has_col = f"has_{cam}"
    video_exists = f"video_{cam}_exists"
    video_path_col = f"video_{cam.lower()}_path"
    roi_col = f"roi_{cam}_path"
    roi_exists = f"roi_{cam}_exists"

    mask_has_cam_video_path_ne_postoji = (df[has_col] == 1) & (df[video_path_col].isna())
    mask_has_cam_video_exists_0 = (df[has_col] == 1) & (df[video_exists] == 0)
    mask_nema_cam_video_exists_1 = (df[has_col] == 0) & (df[video_exists] == 1)
    mask_video_exists_1_roi_ne_postoji = (df[video_exists] == 1) & (df[roi_exists] == 0)
    mask_video_exists_0_roi_postoji = (df[video_exists] == 0) & (df[roi_exists] == 1)

    problematicni = df.loc[
        mask_has_cam_video_path_ne_postoji
        | mask_has_cam_video_exists_0
        | mask_nema_cam_video_exists_1
        | mask_video_exists_1_roi_ne_postoji
        | mask_video_exists_0_roi_postoji,
        [
            "speaker", "sample_name", "language", "split",
            has_col, video_path_col, roi_col, video_exists, roi_exists,
        ],
    ].copy()
    return problematicni


def prepravi_kolonu_kamere(df, cam):
    has_col = f"has_{cam}"
    video_exists_col = f"video_{cam}_exists"
    video_path_col = f"video_{cam.lower()}_path"
    roi_path_col = f"roi_{cam}_path"
    roi_exists_col = f"roi_{cam}_exists"

    df.loc[df[video_path_col].isna(), video_exists_col] = 0
    df.loc[df[roi_path_col].isna(), roi_exists_col] = 0
    df.loc[df[video_exists_col] == 1, has_col] = 1
    df.loc[df[video_exists_col] == 0, has_col] = 0
    df.loc[df[video_exists_col] == 0, video_path_col] = pd.NA
    df.loc[df[video_exists_col] == 0, roi_path_col] = pd.NA
    df.loc[df[video_exists_col] == 0, roi_exists_col] = 0
    return df


def napravi_konzistentne_manifeste(input_ime, output_ime, prefix_izvestaja):
    input_putanja = MANIFESTS_DIR / input_ime
    output_putanja = MANIFESTS_DIR / output_ime
    df = pd.read_csv(input_putanja)
    problematicni_A = izvestaj_za_kameru(df, "A")
    problematicni_L = izvestaj_za_kameru(df, "L")
    problematicni_R = izvestaj_za_kameru(df, "R")
    problematicni_A.to_csv(MANIFESTS_DIR / f"{prefix_izvestaja}_A_problematicni.csv", index=False)
    problematicni_L.to_csv(MANIFESTS_DIR / f"{prefix_izvestaja}_L_problematicni.csv", index=False)
    problematicni_R.to_csv(MANIFESTS_DIR / f"{prefix_izvestaja}_R_problematicni.csv", index=False)

    df = df.dropna(subset=["video_a_path", "video_l_path", "video_r_path"], how="all").copy()
    for cam in ["A", "L", "R"]:
        df = prepravi_kolonu_kamere(df, cam)
    df.to_csv(output_putanja, index=False)
    return df


def main():
    print("Pravljenje pocetnih manifesta")
    napravi_pocetni_manifest()
    print("Pravljenje konzistentnih manifesta")
    napravi_konzistentne_manifeste("manifest_all.csv", "manifest_all_konzistentan.csv", "sanity_all")
    napravi_konzistentne_manifeste("manifest_train.csv", "manifest_train_konzistentan.csv", "sanity_train")
    napravi_konzistentne_manifeste("manifest_test.csv", "manifest_test_konzistentan.csv", "sanity_test")
    print("Gotovo")


if __name__ == "__main__":
    main()
