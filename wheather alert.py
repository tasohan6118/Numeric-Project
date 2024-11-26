import numpy as np
from scipy.linalg import cholesky
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def generate_weather_data(n_regions=5):
    """Simulate weather data for multiple regions."""
    np.random.seed(42)
    A = np.random.rand(n_regions, n_regions)
    A = A @ A.T  # Make A symmetric positive-definite
    b = np.random.rand(n_regions, 1) * 30  # Random temperatures (0-30°C)
    return A, b

def solve_weather_model(A, b):
    """Solve the system using Cholesky decomposition."""
    L = cholesky(A, lower=True)
    y = np.linalg.solve(L, b)  # Forward substitution
    x = np.linalg.solve(L.T, y)  # Back substitution
    return x.flatten()

def plot_weather_data(predictions):
    """Plot the predicted weather data."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(predictions)), predictions, color='skyblue')
    ax.set_xlabel('Region')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Predicted Temperatures by Region')
    plt.tight_layout()
    return fig

def open_file():
    """Open a dialog to select weather data and load it."""
    file_path = filedialog.askopenfilename(
        title="Select Weather Data File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if file_path:
        try:
            data = pd.read_csv(file_path)
            messagebox.showinfo("File Loaded", f"Data loaded from: {file_path}")
            print(data.head())  # Print a preview of the data
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
    else:
        messagebox.showinfo("No File", "No file selected.")

def check_alerts(predictions):
    """Check for temperature alerts and return alert messages."""
    alerts = []
    high_temp_threshold = 30  # Temperature threshold for alert
    for i, temp in enumerate(predictions):
        if temp > high_temp_threshold:
            alerts.append(f"Alert! Region {i + 1}: Temperature exceeds {high_temp_threshold}°C!")
    return alerts

def update_predictions():
    """Generate and display predictions."""
    A, b = generate_weather_data(n_regions=5)
    predictions = solve_weather_model(A, b)

    # Update label with predictions
    predictions_text = "\n".join([f"Region {i + 1}: {temp:.2f}°C"
                                  for i, temp in enumerate(predictions)])
    result_label.config(text=predictions_text)

    # Check for weather alerts
    alerts = check_alerts(predictions)
    if alerts:
        alert_message = "\n".join(alerts)
        messagebox.showwarning("Weather Alerts", alert_message)

    # Plot the results in the GUI
    fig = plot_weather_data(predictions)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=10)

def create_gui():
    """Create the Tkinter GUI for the weather model."""
    global root, result_label

    root = tk.Tk()
    root.title("Weather Prediction Model Using Cholesky Decomposition")

    # Widgets
    ttk.Label(root, text="Weather Prediction Model", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10)
    ttk.Button(root, text="Load Weather Data", command=open_file).grid(row=1, column=0, pady=5)
    ttk.Button(root, text="Generate Predictions", command=update_predictions).grid(row=1, column=1, pady=5)

    result_label = ttk.Label(root, text="", font=("Arial", 12), anchor="w", justify="left")
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
