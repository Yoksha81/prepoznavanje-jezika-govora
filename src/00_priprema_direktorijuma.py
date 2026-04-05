from konfiguracija import (
    DATA_DIR,
    RAW_DATA_DIR,
    WORK_DIR,
    FRAMES_DIR,
    FRAMES_A_DIR,
    FRAMES_L_DIR,
    FRAMES_R_DIR,
    MANIFESTS_DIR,
    FEATURES_DIR,
    RESULTS_DIR,
    DEBUG_DIR,
    TUNING_DIR,
    FINAL_RESULTS_DIR,
)
from funkcije import osiguraj_direktorijum


def main():
    direktorijumi = [
        DATA_DIR,
        RAW_DATA_DIR,
        WORK_DIR,
        FRAMES_DIR,
        FRAMES_A_DIR,
        FRAMES_L_DIR,
        FRAMES_R_DIR,
        MANIFESTS_DIR,
        FEATURES_DIR,
        RESULTS_DIR,
        DEBUG_DIR,
        TUNING_DIR,
        FINAL_RESULTS_DIR,
    ]
    for d in direktorijumi:
        osiguraj_direktorijum(d)
    print("Napravljeni svi direktorijumi.")


if __name__ == "__main__":
    main()
