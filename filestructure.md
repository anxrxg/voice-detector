# Project File Structure – Age & Emotion Voice Detector (Male Only)

project_root/
│
├── main.py                  # Entry point for the app
│
├── models/                  # Pre-trained model files
│   ├── gender_model.pkl
│   ├── age_model.h5
│   ├── emotion_model.h5
│
├── data/                    # Raw & processed datasets
│   ├── gender/
│   ├── age/
│   ├── emotion/
│
├── processed_data/          # Preprocessed audio & feature files
│
├── utils/                   # Helper functions
│   ├── preprocessing.py     # Audio loading & cleaning
│   ├── feature_extraction.py # MFCC & other features
│   ├── model_utils.py       # Model loading, prediction helpers
│
├── training/                # Training scripts
│   ├── train_gender.py
│   ├── train_age.py
│   ├── train_emotion.py
│
├── gui/                     # GUI-related files
│   ├── gui_main.py          # Main GUI interface
│   ├── gui_helpers.py       # GUI utility functions
│
├── tests/                   # Unit tests
│   ├── test_models.py
│   ├── test_preprocessing.py
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── .gitignore               # Ignore unnecessary files in Git

