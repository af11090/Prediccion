import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Cargar el dataset
df = pd.read_csv('OBJ03_DEPURADO.csv', delimiter=',')
# Convertir la columna 'Hmg(g/dl-1)' a numérico
df['Hmg(g/dl-1)'] = df['Hmg(g/dl-1)'].astype(str).str.replace(',', '.', regex=False)
df['Hmg(g/dl-1)'] = pd.to_numeric(df['Hmg(g/dl-1)'])

# Dividir el conjunto de datos en características (X) y la variable objetivo (y)
X = df.drop('Nivel Anemia', axis=1)
y = df['Nivel Anemia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asegúrate de que el número de características es el esperado (14 en este caso)
print(f'Número de características: {X_train.shape[1]}')  # Debería ser 14

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar si los archivos ya existen
if not os.path.exists('random_forest_modelo03.pkl') or not os.path.exists('ramdon_forest_scaler03.pkl'):
    # Entrenar el modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Guardar el modelo y el scaler en archivos
    joblib.dump(rf_model, 'random_forest_modelo03.pkl')
    joblib.dump(scaler, 'ramdon_forest_scaler03.pkl')

# Hacer predicciones y evaluar el modelo (opcional)
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')