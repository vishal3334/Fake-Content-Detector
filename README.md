# Fake Content Detector (Phase 1)

## Setup
    pip install -r requirements.txt

## Task 1 - Train text model (optional but recommended)
Download Kaggle "Fake and Real News Dataset", put Fake.csv and True.csv inside data/
    python text_analyzer.py

## Run the web app (Task 3)
    python app.py
Open http://127.0.0.1:5000

## Files
- text_analyzer.py  : Task 1 - TF-IDF + Logistic Regression
- image_analyzer.py : Task 2 - ELA + EXIF metadata
- app.py + templates/index.html : Task 3 - Flask UI + report
