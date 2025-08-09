import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("restaurant_menu_optimization_data.csv")

# Encoding target (Profitability)
profit_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Profitability'] = df['Profitability'].map(profit_map)

# Pisahkan fitur dan target
X = df[['Price', 'MenuCategory']]
y = df['Profitability']

# One-hot encoding untuk MenuCategory
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['MenuCategory'])
], remainder='passthrough')

# Train-test split (stratifikasi)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pilih model yang ingin diuji
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Training dan evaluasi
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\n===== {name} =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Fungsi untuk plot confusion matrix
def plot_cm(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Plot untuk setiap model
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    plot_cm(name, y_test, y_pred)

import joblib

# Simpan model terbaik
joblib.dump(rf_model, "model.pkl")

# Kalau di preprocessing_restoran.py ada encoder/scaler, simpan juga:
# joblib.dump(encoder, "encoder.pkl")
# joblib.dump(scaler, "scaler.pkl")
