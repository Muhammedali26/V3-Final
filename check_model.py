import pickle
import os

print("Current directory:", os.getcwd())
print("Checking if model file exists:", os.path.exists('nn_model.pkl'))
print("Model file size:", os.path.getsize('nn_model.pkl'))

try:
    with open('nn_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("\nModel info:")
        print("- Model type:", type(model))
        print("- Number of samples:", model.n_samples_fit_)
        print("- Number of neighbors:", model.n_neighbors)
        print("- Algorithm:", model.algorithm)
        print("- Metric:", model.metric)
except Exception as e:
    print("Error loading model:", str(e)) 