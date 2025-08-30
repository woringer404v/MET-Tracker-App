import pickle
import json
import numpy as np

# --- Configuration ---
MODEL_INPUT_FILENAME = 'final_model.pkl'
MODEL_OUTPUT_FILENAME = 'assets/final_model.json'
# This list MUST match the order of features the model was trained on
FEATURE_ORDER = [
    'mean_x', 'std_x', 'min_x', 'max_x', 'range_x',
    'mean_y', 'std_y', 'min_y', 'max_y', 'range_y',
    'mean_z', 'std_z', 'min_z', 'max_z', 'range_z',
    'mean_magnitude', 'std_magnitude',
    'std_jerk_x', 'std_jerk_y', 'std_jerk_z'
]

# Helper to convert numpy types to standard python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    print(f"--- Loading model from '{MODEL_INPUT_FILENAME}' ---")
    with open(MODEL_INPUT_FILENAME, 'rb') as f:
        model = pickle.load(f)

    print("--- Exporting model structure to JSON ---")
    
    # Get the structure of each tree in the forest
    trees_json = []
    for tree_estimator in model.estimators_:
        tree = tree_estimator.tree_
        
        nodes = []
        for i in range(tree.node_count):
            # For leaf nodes, find the majority class
            value = tree.value[i][0]
            predicted_class_index = int(np.argmax(value))

            node_data = {
                "leftChild": tree.children_left[i],
                "rightChild": tree.children_right[i],
                "featureIndex": tree.feature[i],
                "threshold": tree.threshold[i],
                "value": value,
                "predictedClassIndex": predicted_class_index
            }
            nodes.append(node_data)
        
        trees_json.append({"nodes": nodes})

    # Prepare the final JSON object
    model_export = {
        "classes": model.classes_.tolist(),
        "featureOrder": FEATURE_ORDER,
        "trees": trees_json
    }

    # Save to file
    with open(MODEL_OUTPUT_FILENAME, 'w') as f:
        json.dump(model_export, f, default=convert_numpy)

    print(f"\nSuccess! Model exported to '{MODEL_OUTPUT_FILENAME}'")
    print(f"The model has {len(model.classes_)} classes: {model.classes_.tolist()}")
    print(f"The model expects {len(FEATURE_ORDER)} features in a specific order.")

if __name__ == '__main__':
    main()