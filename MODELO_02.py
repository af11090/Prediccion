import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Cargar el dataset
df = pd.read_csv('OBJ02_DEPURADO.csv', delimiter=',')

# Verificar y ajustar nombres de columnas si es necesario
print("Columnas disponibles en el DataFrame:")
print(df.columns)

# Asumamos que el nombre correcto es 'PLT /mm3' con espacio inicial, ajustamos as��
df.columns = df.columns.str.strip()  # Quitar espacios en blanco alrededor de los nombres de las columnas

# Dividir el conjunto de datos en características (X) y la variable objetivo (y)
X = df[['Edad(mes)', 'Peso(kg)', 'Altura(cm)', 'Sexo', 'Hmg(g/dl-1)', 'RBC', 'MCV', 'MCH', 'MCHC', 'RDW']]
y = df[['BTT', 'AIF', 'HbE']]  # Variables objetivo multiclase

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar si los archivos ya existen
if not os.path.exists('random_forest_modelo_btt.pkl') or not os.path.exists('random_forest_modelo_aif.pkl') or not os.path.exists('random_forest_modelo_hbe.pkl') or not os.path.exists('random_forest_scaler02.pkl'):
    # Entrenar un modelo Random Forest para cada variable objetivo
    rf_model_btt = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_aif = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_hbe = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_model_btt.fit(X_train_scaled, y_train['BTT'])
    rf_model_aif.fit(X_train_scaled, y_train['AIF'])
    rf_model_hbe.fit(X_train_scaled, y_train['HbE'])

    # Guardar los modelos y el scaler en archivos
    joblib.dump(rf_model_btt, 'random_forest_modelo_btt.pkl')
    joblib.dump(rf_model_aif, 'random_forest_modelo_aif.pkl')
    joblib.dump(rf_model_hbe, 'random_forest_modelo_hbe.pkl')
    joblib.dump(scaler, 'random_forest_scaler02.pkl')

# Hacer predicciones y evaluar los modelos (opcional)
y_pred_btt_train = rf_model_btt.predict(X_train_scaled)
y_pred_btt_test = rf_model_btt.predict(X_test_scaled)
train_accuracy_btt = accuracy_score(y_train['BTT'], y_pred_btt_train)
test_accuracy_btt = accuracy_score(y_test['BTT'], y_pred_btt_test)

y_pred_aif_train = rf_model_aif.predict(X_train_scaled)
y_pred_aif_test = rf_model_aif.predict(X_test_scaled)
train_accuracy_aif = accuracy_score(y_train['AIF'], y_pred_aif_train)
test_accuracy_aif = accuracy_score(y_test['AIF'], y_pred_aif_test)

y_pred_hbe_train = rf_model_hbe.predict(X_train_scaled)
y_pred_hbe_test = rf_model_hbe.predict(X_test_scaled)
train_accuracy_hbe = accuracy_score(y_train['HbE'], y_pred_hbe_train)
test_accuracy_hbe = accuracy_score(y_test['HbE'], y_pred_hbe_test)

print(f'Train Accuracy BTT: {train_accuracy_btt}')
print(f'Test Accuracy BTT: {test_accuracy_btt}')
print(f'Train Accuracy AIF: {train_accuracy_aif}')
print(f'Test Accuracy AIF: {test_accuracy_aif}')
print(f'Train Accuracy HbE: {train_accuracy_hbe}')
print(f'Test Accuracy HbE: {test_accuracy_hbe}')
