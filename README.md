# Real-Time MET Classification from Smartphone Accelerometer Data

This repository contains the source code, machine learning models, and documentation for a mobile application that performs **real-time classification** of physical activity into four Metabolic Equivalent of Task (MET) classes.

---

## Project Overview

The goal of this project is to provide a tool for personal health monitoring by leveraging the accelerometer sensor present in modern smartphones. The application runs a trained machine learning model entirely **on-device** to classify the user's current activity as **Sedentary**, **Light**, **Moderate**, or **Vigorous**, and tracks the **cumulative time** spent in each category throughout the day.

This is a complete end-to-end system: from a **data collection tool** integrated into the app to a **robust, personalized machine learning model**.

---

## Key Features

- **Tracker Mode**  
  A clean user interface for live activity tracking and viewing daily summary statistics.

- **Collector Mode**  
  A built-in utility to record, label, and manage personal accelerometer data, enabling the creation of custom, high-quality datasets.

- **On-Device Inference**  
  All predictions happen locally, ensuring user **privacy**, **offline functionality**, and **low latency**.

---

## Getting Started: The Android App

You can install the final application on your Android device by downloading the APK file.

### 1) Download the APK
- **Direct Link:**  
  [`https://expo.dev/accounts/amineahmed/projects/MET-Tracker-App/builds/e0d927a2-d8a8-49f7-9469-f5872f758854`](#)  


- **QR Code:**  
  Scan the QR code below with your phone’s camera to start the download.  
  
  ![Download via QR](images/qr_code.png)

### 2) Installation Instructions

Since this app is not downloaded from the Google Play Store, you may need to enable **“Install from unknown sources”** in your phone’s settings.

1. Download the `.apk` file using one of the methods above.  
2. Open your phone’s file manager and navigate to **Downloads**.  
3. Tap on the downloaded `.apk` file.  
4. If you see a warning about unknown sources, tap **Settings** and enable the permission for your file manager or browser.  
5. Go back and tap **Install**.  
6. Once installed, the app will appear on your home screen.

---

## The Machine Learning Model

- **Final model:** `final_model.pkl`  
- **Algorithm:** Random Forest Classifier  
- **Training data:** Exclusively on a **personalized** dataset collected by the developer (to match device-specific sensor characteristics and user movement patterns).  
- **Features:** 19 engineered features derived from **windowed accelerometer** data, augmented via **Jittering**, **Scaling**, and **Rotation** to increase robustness.

---

## Reproducibility: Train Your Own Model

The full training process is reproducible. The repository includes two Python scripts:

### Training Scripts

- `train.py` — **Baseline Model**  
  Trains a model on raw, personally collected data **without** augmentation. It follows a UCI-inspired pipeline: trimming, smoothing, windowing, and feature extraction. Useful for a baseline metric.

- `train_augmented.py` — **Final Model**  
  Generates `final_model.pkl`. Implements the full data pipeline with augmentation (**Jittering**, **Scaling**, **Rotation**) for a more robust and generalized model.

### How to Run the Training

**Prerequisites**  
Make sure you have Python 3 and the required libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

**Data**
Place your personally collected `.csv` recording files into the folder `my_recordings` located in the same directory as the training scripts.

**Execute**
Run the desired training script from your terminal. For the final model:

```bash
python train_augmented.py
```

The script will process the data, train the model, display evaluation plots (Feature Importance and Confusion Matrix), and save the final model file.
