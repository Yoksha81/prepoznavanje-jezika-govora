import pandas as pd

from konfiguracija import (
    INPUT_MANIFEST,
    CROP_MANIFEST,
    FRAMES_BY_CAMERA,
    N_FRAMES,
    ROI_PADDING,
    ALIGN_SCALE,
)
from funkcije import (
    osiguraj_direktorijum,
    ucitaj_manifest,
    interval_govora_iz_align,
    ekvidistantne_tacke,
    vremena_u_frejmove,
    ucitaj_roi_fajl,
    nadji_roi_za_frejm,
    otvori_video,
    procitaj_frejm,
    iseci_roi,
    sacuvaj_sliku,
)

VIEWS = ["A", "L", "R"]


def obradi_uzorak_i_kameru(uzorak, view):
    video_exists_col = f"video_{view}_exists"
    roi_exists_col = f"roi_{view}_exists"
    video_path_col = f"video_{view.lower()}_path"
    roi_path_col = f"roi_{view}_path"

    if uzorak[video_exists_col] != 1 or uzorak[roi_exists_col] != 1 or uzorak["align_exists"] != 1:
        return []

    start_s, end_s = interval_govora_iz_align(uzorak["align_path"], ALIGN_SCALE)
    if start_s is None or end_s is None:
        return []

    cap_data = otvori_video(uzorak[video_path_col])
    if cap_data is None:
        return []
    cap, fps, frame_count = cap_data
    roi_df = ucitaj_roi_fajl(uzorak[roi_path_col])

    vremena_s = ekvidistantne_tacke(start_s, end_s, N_FRAMES, mode="centers")
    frame_indices = vremena_u_frejmove(vremena_s, fps, frame_count)

    out_dir = FRAMES_BY_CAMERA[view] / uzorak["speaker"] / uzorak["sample_name"]
    osiguraj_direktorijum(out_dir)

    records = []
    for redni_broj, (time_s, frame_idx) in enumerate(zip(vremena_s, frame_indices)):
        frame = procitaj_frejm(cap, frame_idx)
        if frame is None:
            continue
        roi = nadji_roi_za_frejm(roi_df, frame_idx)
        if roi is None:
            continue
        x, y, w, h = roi
        crop = iseci_roi(frame, roi, padding=ROI_PADDING)
        if crop is None:
            continue
        naziv = f"{uzorak['sample_name']}_{view}_f{int(frame_idx):04d}_{redni_broj:02d}.png"
        out_path = out_dir / naziv
        sacuvaj_sliku(out_path, crop)
        orig_h, orig_w = crop.shape[:2]
        records.append({
            "speaker": uzorak["speaker"],
            "sample_name": uzorak["sample_name"],
            "language": uzorak["language"],
            "split": uzorak["split"],
            "camera": view,
            "time_sec": float(time_s),
            "frame_idx": int(frame_idx),
            "crop_order": int(redni_broj),
            "crop_path": str(out_path),
            "video_path": uzorak[video_path_col],
            "roi_path": uzorak[roi_path_col],
            "align_path": uzorak["align_path"],
            "fps": float(fps),
            "frame_count": int(frame_count),
            "speech_start_s": float(start_s),
            "speech_end_s": float(end_s),
            "roi_x": int(x),
            "roi_y": int(y),
            "roi_w": int(w),
            "roi_h": int(h),
            "roi_area": int(w * h),
            "roi_aspect_ratio": float(w / h) if h else None,
            "orig_crop_w": int(orig_w),
            "orig_crop_h": int(orig_h),
        })
    cap.release()
    return records


def main():
    df = ucitaj_manifest(INPUT_MANIFEST)
    svi_cropovi = []
    for _, uzorak in df.iterrows():
        for view in VIEWS:
            svi_cropovi.extend(obradi_uzorak_i_kameru(uzorak, view))
    crop_df = pd.DataFrame(svi_cropovi)
    crop_df.to_csv(CROP_MANIFEST, index=False)
    print(f"Sacuvan crop manifest: {CROP_MANIFEST}")
    print(f"Broj sacuvanih cropova: {len(crop_df)}")


if __name__ == "__main__":
    main()
