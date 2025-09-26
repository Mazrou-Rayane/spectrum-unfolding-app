import os
import zipfile
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from flask import Flask, request, send_file, render_template, redirect, url_for, jsonify

app = Flask(__name__)

# --------------------------
# 1) Extract Models
# --------------------------
if not os.path.exists("models"):
    with zipfile.ZipFile("xgb_models.zip", "r") as zip_ref:
        zip_ref.extractall("models")
    print("📦 Extracted models from xgb_models.zip to models")

# --------------------------
# 2) Load Models
# --------------------------
models = []
for i in range(len([f for f in os.listdir("models") if f.endswith(".json")])):
    booster = xgb.Booster()
    model_path = os.path.join("models", f"xgb_model_{i}.json")
    booster.load_model(model_path)
    models.append(booster)

print(f"✅ Loaded {len(models)} models.")

# --------------------------
# 3) Globals
# --------------------------
last_output = "predictions.csv"
last_predictions = None  # Store predictions for plotting

# --------------------------
# 4) Helper Functions
# --------------------------

# Energy bin edges in eV
bin_edges = np.array([
    0.001, 0.00126, 0.00158, 0.002, 0.00251, 0.00316, 0.00398, 0.00501, 0.00631, 0.00794,
    0.01, 0.0126, 0.0158, 0.02, 0.0251, 0.0316, 0.0398, 0.0501, 0.0631, 0.0794,
    0.1, 0.126, 0.158, 0.2, 0.251, 0.316, 0.398, 0.501, 0.631, 0.794,
    1, 1.26, 1.58, 2, 2.51, 3.16, 3.98, 5.01, 6.31, 7.94,
    10, 12.6, 15.8, 20, 25.1, 31.6, 39.8, 50.1, 63.1, 79.4,
    100, 126, 158, 200, 251, 316, 398, 501, 631, 794,
    1000, 1260, 1580, 2000, 2510, 3160, 3980, 5010, 6310, 7940,
    10000, 12600, 15800, 20000, 25100, 31600, 39800, 50100, 63100, 79400,
    100000, 126000, 158000, 200000, 251000, 316000, 398000, 501000, 631000, 794000,
    1000000, 1260000, 1580000, 2000000, 2510000, 3160000, 3980000, 5010000, 6310000, 7940000,
    10000000, 12600000, 15800000, 20000000
])

def create_spectrum_plot(predictions, sample_idx=0, title_suffix=""):
    """Create a spectrum plot for a single sample"""
    try:
        print(f"Creating plot for sample {sample_idx}")
        plt.figure(figsize=(14, 8))
        
        print(f"Bin edges shape: {len(bin_edges)}")
        
        # Fix dimension mismatch - use only the number of predictions available
        num_predictions = len(predictions[sample_idx])
        print(f"Number of predictions: {num_predictions}")
        
        # Use the bin edges directly, truncated to match predictions
        x_values = bin_edges[:num_predictions]
        
        if num_predictions != len(bin_edges):
            print(f"Info: Using first {num_predictions} bin edges to match predictions")
        
        # Check for any invalid values
        sample_data = predictions[sample_idx]
        if np.any(sample_data <= 0):
            print("Warning: Found non-positive values, setting minimum to 1e-10")
            sample_data = np.maximum(sample_data, 1e-10)
        
        plt.semilogx(x_values, sample_data, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
        plt.fill_between(x_values, sample_data, alpha=0.3)
        
        plt.title(f'Predicted Energy Spectrum - Sample {sample_idx + 1}{title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel('Energy (eV)', fontsize=14)
        plt.ylabel('Predicted Flux', fontsize=14)
        plt.grid(True, alpha=0.3, which='both')
        
        # Improve y-axis formatting for linear scale
        ax = plt.gca()
        
        # Set linear y-axis ticks
        y_min, y_max = sample_data.min(), sample_data.max()
        
        # Create evenly spaced tick positions for linear scale
        y_range = y_max - y_min
        y_ticks = np.linspace(y_min, y_max, num=6)
        ax.set_yticks(y_ticks)
        
        # Format y-axis labels for linear scale
        ax.set_yticklabels([f'{y:.4f}' for y in y_ticks])
        
        # Set y-axis limits with some padding
        padding = y_range * 0.1
        plt.ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=120, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        print("Plot created successfully")
        return plot_url
        
    except Exception as e:
        plt.close()  # Make sure to close any open figures
        print(f"Error in create_spectrum_plot: {str(e)}")
        raise e

def create_all_spectra_plot(predictions, max_samples=10):
    """Create an overlay plot of multiple spectra"""
    plt.figure(figsize=(14, 8))
    
    num_samples = min(len(predictions), max_samples)
    x_values = np.arange(1, predictions.shape[1] + 1)
    
    # Create a colormap for different samples
    colors = plt.cm.Set3(np.linspace(0, 1, num_samples))
    
    for i in range(num_samples):
        plt.plot(x_values, predictions[i], linewidth=2, alpha=0.7, 
                label=f'Sample {i+1}', color=colors[i])
    
    plt.title(f'Overlay of {num_samples} Predicted Spectra', fontsize=14, fontweight='bold')
    plt.xlabel('Spectrum Bin', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_statistics_plot(predictions):
    """Create statistical plots (mean, std, etc.)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x_values = np.arange(1, predictions.shape[1] + 1)
    
    # Mean spectrum
    mean_spectrum = np.mean(predictions, axis=0)
    ax1.plot(x_values, mean_spectrum, 'r-', linewidth=2)
    ax1.fill_between(x_values, mean_spectrum, alpha=0.3, color='red')
    ax1.set_title('Mean Predicted Spectrum')
    ax1.set_xlabel('Spectrum Bin')
    ax1.set_ylabel('Mean Value')
    ax1.grid(True, alpha=0.3)
    
    # Standard deviation
    std_spectrum = np.std(predictions, axis=0)
    ax2.plot(x_values, std_spectrum, 'g-', linewidth=2)
    ax2.fill_between(x_values, std_spectrum, alpha=0.3, color='green')
    ax2.set_title('Standard Deviation Across Samples')
    ax2.set_xlabel('Spectrum Bin')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True, alpha=0.3)
    
    # Min and Max
    min_spectrum = np.min(predictions, axis=0)
    max_spectrum = np.max(predictions, axis=0)
    ax3.plot(x_values, min_spectrum, 'b-', linewidth=2, label='Min')
    ax3.plot(x_values, max_spectrum, 'orange', linewidth=2, label='Max')
    ax3.fill_between(x_values, min_spectrum, max_spectrum, alpha=0.3)
    ax3.set_title('Min/Max Envelope')
    ax3.set_xlabel('Spectrum Bin')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distribution of total intensity per sample
    total_intensities = np.sum(predictions, axis=1)
    ax4.hist(total_intensities, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('Distribution of Total Spectrum Intensity')
    ax4.set_xlabel('Total Intensity')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# --------------------------
# 5) Routes
# --------------------------
@app.route("/")
def index():
    return render_template("index.html", models=models)

@app.route("/predict", methods=["POST"])
def predict():
    global last_output, last_predictions
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Load test CSV
    test_df = pd.read_csv(file)
    count_cols = [c for c in test_df.columns if c.startswith("count_")]

    # Normalize counts
    counts_total = test_df[count_cols].sum(axis=1)
    counts_total[counts_total == 0] = 1
    test_df[count_cols] = test_df[count_cols].div(counts_total, axis=0)

    X_test = test_df[count_cols].values
    dtest = xgb.DMatrix(X_test)

    # Predict with each model (one per spectrum bin)
    predictions = []
    for booster in models:
        predictions.append(booster.predict(dtest))
    predictions = np.column_stack(predictions)
    
    # Store predictions for plotting
    last_predictions = predictions

    # Save predictions
    pred_df = pd.DataFrame(predictions, columns=[f"spectrum_value_{i+1}" for i in range(predictions.shape[1])])
    last_output = "predictions.csv"
    pred_df.to_csv(last_output, index=False)

    # Send preview (first 5 rows)
    preview = pred_df.head(5).round(4)

    return render_template("index.html", preview=preview, models=models, 
                         num_samples=len(predictions), show_plots=True)

@app.route("/plot/<int:sample_idx>")
def plot_single(sample_idx):
    """Generate plot for a single sample"""
    try:
        if last_predictions is None:
            return jsonify({"error": "No predictions available"})
        
        if sample_idx >= len(last_predictions):
            return jsonify({"error": "Sample index out of range"})
        
        print(f"Generating plot for sample {sample_idx}")
        print(f"Predictions shape: {last_predictions.shape}")
        print(f"Sample data shape: {last_predictions[sample_idx].shape}")
        
        plot_url = create_spectrum_plot(last_predictions, sample_idx)
        return jsonify({"plot": plot_url})
        
    except Exception as e:
        print(f"Error in plot_single: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating plot: {str(e)}"})

@app.route("/plot/overlay")
def plot_overlay():
    """Generate overlay plot of multiple samples"""
    if last_predictions is None:
        return jsonify({"error": "No predictions available"})
    
    plot_url = create_all_spectra_plot(last_predictions)
    return jsonify({"plot": plot_url})

@app.route("/plot/statistics")
def plot_statistics():
    """Generate statistical plots"""
    if last_predictions is None:
        return jsonify({"error": "No predictions available"})
    
    plot_url = create_statistics_plot(last_predictions)
    return jsonify({"plot": plot_url})

@app.route("/download")
def download():
    return send_file(last_output, as_attachment=True)

# --------------------------
# 6) Run
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)