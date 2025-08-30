import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# --- 1. Configuration ---
DATA_DIR = 'my_recordings'
FINAL_MODEL_FILENAME = 'model.pkl'

# Calculation: 2.56 seconds * 20 Hz = 51.2 samples. We'll use 51.
WINDOW_SIZE = 51
OVERLAP_PERCENTAGE = 0.5  # 50% overlap
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
if STEP_SIZE == 0: STEP_SIZE = 1 # Ensure step size is at least 1 for small windows

# --- 2. Data Loading Function ---
def load_all_recordings(directory):
    all_files = []
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_files.append(df)
    if not all_files: return pd.DataFrame()
    return pd.concat(all_files, ignore_index=True)

# --- 3. Feature Engineering Function ---
def extract_features_from_window(window):
    """
    This function engineers a rich set of features from a single window of sensor data.
    """
    features = {}
    
    # --- Basic Statistical Features ---
    for axis in ['x', 'y', 'z']:
        signal = window[axis]
        features[f'mean_{axis}'] = signal.mean()
        features[f'std_{axis}'] = signal.std()
        features[f'min_{axis}'] = signal.min()
        features[f'max_{axis}'] = signal.max()
        # --- Engineered Feature 1: Range ---
        # Captures the amplitude or "swing" of the motion on each axis.
        features[f'range_{axis}'] = signal.max() - signal.min()

    # --- Engineered Feature 2: Magnitude ---
    # Creates an orientation-independent measure of total acceleration.
    magnitude = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    features['mean_magnitude'] = magnitude.mean()
    features['std_magnitude'] = magnitude.std()

    # --- Engineered Feature 3: Jerk (Rate of change of acceleration) ---
    # Represents the "smoothness" or "jerkiness" of the motion.
    for axis in ['x', 'y', 'z']:
        jerk = np.diff(window[axis], n=1) # First derivative (change between samples)
        features[f'std_jerk_{axis}'] = np.std(jerk)

    return features

# --- 4. Main Data Processing Pipeline ---
def process_data_pipeline(df):
    feature_data = []

    # Trim start/end of each session
    def trim_sessions(df, trim_ms=3000):
        trimmed_dfs = []
        for session_id, group in df.groupby('session_id'):
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            time_mask = (group['timestamp'] > start_time + trim_ms) & (group['timestamp'] < end_time - trim_ms)
            trimmed_dfs.append(group[time_mask])
        return pd.concat(trimmed_dfs, ignore_index=True)

    processed_df = trim_sessions(df)
    
    for session_id, session_df in processed_df.groupby('session_id'):
        # Smooth signals with a moving average
        session_df[['x', 'y', 'z']] = session_df[['x', 'y', 'z']].rolling(window=5, min_periods=1, center=True).mean()

        for i in range(0, len(session_df) - WINDOW_SIZE + 1, STEP_SIZE):
            window = session_df.iloc[i:i + WINDOW_SIZE]
            if window.empty: continue
            
            features = extract_features_from_window(window)
            features['label'] = window['label'].iloc[0]
            feature_data.append(features)

    return pd.DataFrame(feature_data).dropna()


# --- Main Execution ---
def main():
    print("--- Starting Model Training ---")
    
    # 1. Load and Process Data
    raw_df = load_all_recordings(DATA_DIR)
    if raw_df.empty:
        print("\nTraining stopped. No data found.")
        return
        
    print(f"\n1. Processing {raw_df['session_id'].nunique()} recordings...")
    features_df = process_data_pipeline(raw_df)
    print(f"   -> Generated {features_df.shape[0]} windows with {features_df.shape[1]-1} features each.")

    # 2. Prepare for Training
    X = features_df.drop('label', axis=1)
    y = features_df['label']
    
    # Stratified split to ensure class balance in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\n2. Splitting data: {len(X_train)} for training, {len(X_test)} for testing.")

    # 3. Train the Model
    print("\n3. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   -> Model training complete.")

    # 4. Evaluate the Model
    print("\n4. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy on Test Set: {accuracy * 100:.2f}%")

    # --- 5. Feature Selection via Importance ---
    print("\n5. Analyzing feature importance...")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances, y=importances.index, palette='viridis')
    plt.title('Feature Importances for Model', fontsize=18)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # --- 6. Confusion Matrix ---
    print("\n6. Visualizing Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 7. Save the Final Model
    print(f"\n7. Saving final model to '{FINAL_MODEL_FILENAME}'...")
    with open(FINAL_MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    print("   -> Model saved successfully.")
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()