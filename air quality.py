
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

class AirQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Air Quality Prediction System")
        self.root.geometry("1200x800")
        self.bg_color = '#F0F8FF'  # Light cyan background color
        self.root.configure(bg=self.bg_color)

        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

        self.aqi_categories = {
            (0, 50): ('Good', '#55A84F'),
            (51, 100): ('Moderate', '#F4D03F'),
            (101, 150): ('Unhealthy for Sensitive', '#F39C12'),
            (151, 200): ('Unhealthy', '#E74C3C'),
            (201, 300): ('Very Unhealthy', '#C0392B'),
            (301, 500): ('Hazardous', '#8E44AD')
        }

        self.default_dataset = "AirQuality.csv"
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('Header.TLabel', background=self.bg_color, font=('Helvetica', 16, 'bold'))

        main_frame = ttk.Frame(self.root, padding="20", style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_frame, text="Air Quality Prediction System", style='Header.TLabel')
        header.pack(pady=10)

        data_frame = ttk.LabelFrame(main_frame, text="Data Loading", padding="15")
        data_frame.pack(fill=tk.X, pady=10, ipadx=10, ipady=5)

        ttk.Label(data_frame, text="Dataset Path:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.file_entry = ttk.Entry(data_frame, width=50)
        self.file_entry.insert(0, self.default_dataset)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(data_frame, text="Load & Analyze Data", command=self.load_data).grid(row=1, column=0, columnspan=3, pady=10)

        pred_frame = ttk.LabelFrame(main_frame, text="Make Prediction", padding="15")
        pred_frame.pack(fill=tk.X, pady=10, ipadx=10, ipady=5)

        features = [
            'CO_GT_var', 'PT08_S1_CO_var', 'NMHC_GT_var',
            'C6H6_GT_var', 'PT08_S2_NMHC_var', 'NOx_GT_var'
        ]

        for i, feature in enumerate(features):
            label = feature.replace("_", " ").replace("var", "(input)")
            ttk.Label(pred_frame, text=f"{label}:").grid(row=i//3, column=(i%3)*2, padx=5, pady=5, sticky=tk.W)
            var = tk.DoubleVar()
            setattr(self, feature, var)
            ttk.Entry(pred_frame, textvariable=var, width=10).grid(row=i//3, column=(i%3)*2+1, padx=5, pady=5, sticky=tk.W)

        ttk.Button(pred_frame, text="Predict AQI", command=self.predict_aqi).grid(row=2, column=0, columnspan=6, pady=10)

        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.results_text = tk.Text(left_panel, wrap=tk.WORD, height=15, font=('Helvetica', 10), bg='#ffffff')
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        aqi_display_frame = ttk.Frame(left_panel)
        aqi_display_frame.pack(fill=tk.X, pady=10)

        self.aqi_value = ttk.Label(aqi_display_frame, text="Predicted AQI: -", font=('Helvetica', 24), background=self.bg_color)
        self.aqi_value.pack()

        self.aqi_category = ttk.Label(aqi_display_frame, text="Category: -", font=('Helvetica', 18), background=self.bg_color)
        self.aqi_category.pack()

        self.health_advice = tk.Message(left_panel, text="Health Advice: -", font=('Helvetica', 12), width=400, bg='#ffffcc')
        self.health_advice.pack(fill=tk.X, pady=10)

        viz_frame = ttk.Frame(results_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=(8, 6), dpi=100, facecolor='#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, background='#d9edf7')
        self.status_bar.pack(fill=tk.X)

    def browse_file(self):
        filepath = filedialog.askopenfilename(title="Select Air Quality Dataset", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialfile=self.default_dataset)
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)

    def load_data(self):
        filepath = self.file_entry.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a dataset file")
            return
        try:
            self.df = pd.read_csv(filepath, sep=';', decimal=',')
            self.df.columns = [col.strip() for col in self.df.columns]

            if 'Date' in self.df.columns and 'Time' in self.df.columns:
                try:
                    self.df['DateTime'] = pd.to_datetime(self.df['Date'].astype(str).str.strip() + ' ' + self.df['Time'].astype(str).str.strip(), errors='coerce', dayfirst=True)
                except Exception as e:
                    messagebox.showwarning("DateTime Parsing Error", f"Failed to parse DateTime: {str(e)}")

            self.df.replace(-200, np.nan, inplace=True)

            features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)']
            numeric_cols = self.df[features]
            X_imputed = self.imputer.fit_transform(numeric_cols)
            self.df[features] = X_imputed

            self.df['AQI'] = self.calculate_aqi(self.df)

            if self.df['DateTime'].isnull().all():
                self.df.drop(columns=['DateTime'], inplace=True)

            X = self.df[features]
            y = self.df['AQI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Dataset loaded with {len(self.df)} records\n")
            if 'DateTime' in self.df.columns:
                self.results_text.insert(tk.END, f"Date range: {self.df['DateTime'].min().date()} to {self.df['DateTime'].max().date()}\n")
            self.results_text.insert(tk.END, "\nModel Evaluation:\n")
            self.results_text.insert(tk.END, f"Mean Absolute Error: {mae:.2f}\n")
            self.results_text.insert(tk.END, f"R-squared Score: {r2:.2f}\n\n")
            self.results_text.insert(tk.END, "Feature Importance:\n")
            for feature, importance in zip(features, self.model.feature_importances_):
                self.results_text.insert(tk.END, f"{feature}: {importance:.3f}\n")

            self.plot_aqi_trend()
            self.status_var.set("Data loaded and model trained successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")

    def calculate_aqi(self, df):
        weights = {
            'CO(GT)': 0.25,
            'PT08.S1(CO)': 0.15,
            'NMHC(GT)': 0.10,
            'C6H6(GT)': 0.15,
            'PT08.S2(NMHC)': 0.15,
            'NOx(GT)': 0.20
        }
        return sum(df[col] * weight for col, weight in weights.items())

    def get_aqi_category(self, aqi_value):
        for (min_val, max_val), (category, color) in self.aqi_categories.items():
            if min_val <= aqi_value <= max_val:
                advice = {
                    'Good': "Air quality is satisfactory. Enjoy outdoor activities.",
                    'Moderate': "Acceptable air quality. Sensitive individuals should reduce prolonged exertion.",
                    'Unhealthy for Sensitive': "People with respiratory or heart conditions should reduce outdoor activities.",
                    'Unhealthy': "Everyone may begin to experience health effects. Limit outdoor activities.",
                    'Very Unhealthy': "Health alert - everyone may experience serious health effects.",
                    'Hazardous': "Health warning of emergency conditions - avoid all outdoor activities."
                }.get(category, "No specific health advice available.")
                return category, color, advice
        return "Unknown", "gray", "No health advice available"

    def predict_aqi(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load data and train model first")
            return
        try:
            input_values = []
            features = [
                'CO_GT_var', 'PT08_S1_CO_var', 'NMHC_GT_var',
                'C6H6_GT_var', 'PT08_S2_NMHC_var', 'NOx_GT_var'
            ]
            for feature in features:
                var = getattr(self, feature)
                value = var.get()
                if value < 0:
                    raise ValueError("Pollutant values cannot be negative")
                input_values.append(value)

            X_new = np.array([input_values])
            X_new_scaled = self.scaler.transform(X_new)
            aqi_pred = self.model.predict(X_new_scaled)[0]
            category, color, advice = self.get_aqi_category(aqi_pred)

            self.aqi_value.config(text=f"Predicted AQI: {aqi_pred:.1f}")
            self.aqi_category.config(text=f"Category: {category}", foreground=color)
            self.health_advice.config(text=f"Health Advice: {advice}")

            self.results_text.insert(tk.END, "\n\nPrediction Results:\n")
            display_features = [
                'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)',
                'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)'
            ]
            for feature, value in zip(display_features, input_values):
                self.results_text.insert(tk.END, f"{feature}: {value}\n")
            self.results_text.insert(tk.END, f"Predicted AQI: {aqi_pred:.1f} ({category})\n")

            self.status_var.set("Prediction completed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Error during prediction")

    def plot_aqi_trend(self):
        
        if self.df is None or 'DateTime' not in self.df.columns:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor='#f0f0f0')
        daily_aqi = self.df.set_index('DateTime')['AQI'].resample('D').mean()
        for (min_val, max_val), (category, color) in self.aqi_categories.items():
            subset = daily_aqi[(daily_aqi >= min_val) & (daily_aqi <= max_val)]
            if not subset.empty:
                ax.plot(subset.index, subset.values, 'o', color=color, label=category, markersize=5)
        ax.set_title('Daily Average AQI Over Time', pad=20)
        ax.set_xlabel('Date', labelpad=10)
        ax.set_ylabel('AQI', labelpad=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AirQualityApp(root)
    root.mainloop()
 