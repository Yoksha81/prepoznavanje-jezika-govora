00_priprema_direktorijuma.py
    - pravi osnovne foldere projekta
    - frames/A, frames/L i frames/R se kreiraju ovde
    - results/model_tuning i results/final_models se kreiraju ovde

01_manifesti.py
    - pravi pocetne manifeste iz baze
    - radi sanity check
    - pravi konzistentne/fiksirane manifeste
    - cuva CSV izvestaje o problemima

02_crop_manifest.py
    - iz anonimisanih videa i ROI fajlova izvlaci 40 cropova po uzorku
    - cuva cropove direktno u frames/A, frames/L i frames/R
    - pravi crop manifest, bez dodatnog original_crops sloja

03_hog_obelezja.py
    - iz crop manifesta pravi HOG video-level obelezja po kamerama

04_resnet18_obelezja.py
    - iz crop manifesta pravi ResNet-18 video-level obelezja po kamerama

05_tuning_modela.py
    - ima dva rezima rada:
      1) "puni" grid search za sve kombinacije
      2) "refinement" koji dopisuje rezultate u postojece CSV fajlove
    - rezultate cuva u results/model_tuning

06_finalna_evaluacija.py
    - trenira finalne modele na train speakerima
    - evaluira na test speakerima
    - radi homogeni fusion i strict best-of-best fusion
    - cuva metrike, predikcije i matrice konfuzije u results/final_models

zajednicki moduli:
konfiguracija.py
funkcije.py

Preporuceno okruzenje:
- Python 3.10
- Projekat je testiran na lokalnom CPU okruzenju
- Pre pokretanja potrebno je postaviti sirove podatke u data/raw
- Generisani izlazi se cuvaju u work/
