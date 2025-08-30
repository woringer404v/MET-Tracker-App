import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from scipy.spatial.transform import Rotation as R

# --- 1. Configuration ---
DATA_DIR = 'my_recordings'
FINAL_MODEL_FILENAME = 'final_model.pkl'
WINDOW_SIZE = 51  # ~2.56 seconds at 20Hz
OVERLAP_PERCENTAGE = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
if STEP_SIZE == 0: STEP_SIZE = 1

# --- 2. Data Augmentation Functions ---

def jitter(signal_window, sigma=0.03):
    """Adds random Gaussian noise to the signal."""
    noise = np.random.normal(loc=0, scale=sigma, size=signal_window.shape)
    return signal_window + noise

def scale(signal_window, factor_range=(0.9, 1.1)):
    """Scales the signal's intensity."""
    scaling_factor = np.random.uniform(low=factor_range[0], high=factor_range[1])
    return signal_window * scaling_factor

def rotate(signal_window):
    """Applies a random 3D rotation to the [x, y, z] vectors."""
    # Create a random rotation matrix
    random_rotation = R.from_euler('xyz', np.random.uniform(0, 360, size=3), degrees=True)
    # Apply the rotation to each [x, y, z] point in the window
    return random_rotation.apply(signal_window)

# --- 3. Feature Engineering Function ---
def extract_features_from_window(window_df):
    features = {}
    # Basic Stats
    for axis in ['x', 'y', 'z']:
        signal = window_df[axis]
        features[f'mean_{axis}'] = signal.mean()
        features[f'std_{axis}'] = signal.std()
        features[f'min_{axis}'] = signal.min()
        features[f'max_{axis}'] = signal.max()
        features[f'range_{axis}'] = signal.max() - signal.min()
    # Magnitude Features
    magnitude = np.sqrt(window_df['x']**2 + window_df['y']**2 + window_df['z']**2)
    features['mean_magnitude'] = magnitude.mean()
    features['std_magnitude'] = magnitude.std()
    # Jerk Features
    for axis in ['x', 'y', 'z']:
        jerk = np.diff(window_df[axis], n=1)
        features[f'std_jerk_{axis}'] = np.std(jerk)
    return features

# --- 4. Main Data Processing Pipeline ---
def process_data_pipeline(df):
    feature_data = []
    
    # Trim sessions
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
        # Smooth signals
        session_df[['x', 'y', 'z']] = session_df[['x', 'y', 'z']].rolling(window=5, min_periods=1, center=True).mean()

        for i in range(0, len(session_df) - WINDOW_SIZE + 1, STEP_SIZE):
            original_window = session_df.iloc[i:i + WINDOW_SIZE]
            if len(original_window) < WINDOW_SIZE: continue
            
            label = original_window['label'].iloc[0]
            
            # --- AUGMENTATION STEP ---
            windows_to_process = {'original': original_window}
            
            # Extract raw xyz data for augmentation
            xyz_original = original_window[['x', 'y', 'z']].values

            # Create augmented versions
            windows_to_process['jittered'] = pd.DataFrame(jitter(xyz_original), columns=['x', 'y', 'z'])
            windows_to_process['scaled'] = pd.DataFrame(scale(xyz_original), columns=['x', 'y', 'z'])
            windows_to_process['rotated'] = pd.DataFrame(rotate(xyz_original), columns=['x', 'y', 'z'])
            
            # Extract features from original and all augmented windows
            for key, window in windows_to_process.items():
                features = extract_features_from_window(window)
                features['label'] = label
                feature_data.append(features)

    return pd.DataFrame(feature_data).dropna()

# --- Main Execution ---
def main():
    print("--- Starting Augmented Model Training ---")
    
    # 1. Load Data
    raw_df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in os.listdir(DATA_DIR) if f.endswith('.csv')], ignore_index=True)
    if raw_df.empty:
        print("\nTraining stopped. No data found.")
        return
        
    # 2. Process Data with Augmentation
    print(f"\n1. Processing {raw_df['session_id'].nunique()} recordings with data augmentation...")
    features_df = process_data_pipeline(raw_df)
    print(f"   -> Generated {features_df.shape[0]} total windows (original + augmented).")
    print(f"   -> Each window has {features_df.shape[1]-1} features.")

    # 3. Prepare for Training
    X = features_df.drop('label', axis=1)
    y = features_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\n2. Splitting data: {len(X_train)} for training, {len(X_test)} for testing.")

    # 4. Train the Model
    print("\n3. Training Random Forest Classifier on augmented data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   -> Model training complete.")

    # 5. Evaluate the Model
    print("\n4. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    # 6. Show and Plot Feature Importances
    print("\n5. Analyzing feature importances...")
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance from Random Forest', fontsize=18)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # 7. Show Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix (on Augmented Data)', fontsize=18)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 8. Save the Final Model
    print(f"\n6. Saving final augmented model to '{FINAL_MODEL_FILENAME}'...")
    with open(FINAL_MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    print("   -> Model saved successfully.")
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()