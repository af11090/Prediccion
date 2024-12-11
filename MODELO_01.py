import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# CARGAR EL DATASET
df = pd.read_csv('OBJ01_DEPURADO.csv', delimiter=',')

# Mostrar las primeras 3 filas del DataFrame
print(df.head(3))

# Convertir la columna 'Hmg(g/dl-1)' a numérico
df['Hmg(g/dl-1)'] = df['Hmg(g/dl-1)'].astype(str).str.replace(',', '.', regex=False)
df['Hmg(g/dl-1)'] = pd.to_numeric(df['Hmg(g/dl-1)'])

# Mostrar las primeras 3 filas del DataFrame
print(df.head(3))

# Dividir el conjunto de datos en características (X) y la variable objetivo (y)
X = df.drop('Anemia', axis=1)
y = df['Anemia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar si los archivos ya existen
if not os.path.exists('random_forest_modelo01.pkl') or not os.path.exists('ramdon_forest_scaler01.pkl'):
    # Entrenar el modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Guardar el modelo y el scaler en archivos
    joblib.dump(rf_model, 'random_forest_modelo01.pkl')
    joblib.dump(scaler, 'ramdon_forest_scaler01.pkl')

# Hacer predicciones y evaluar el modelo (opcional)
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')