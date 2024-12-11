from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import os
import subprocess

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Función para cargar modelos y scalers si existen
def cargar_modelo_y_scaler(modelo_path, scaler_path, script_path):
    if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
        subprocess.run(['python', script_path], check=True)
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    return modelo, scaler

# Cargar los modelos y los scalers
modelo1, scaler1 = cargar_modelo_y_scaler('random_forest_modelo01.pkl', 'ramdon_forest_scaler01.pkl', 'MODELO_01.py')
modelo_btt, _ = cargar_modelo_y_scaler('random_forest_modelo_btt.pkl', 'random_forest_scaler02.pkl', 'MODELO_02.py')
modelo_aif, _ = cargar_modelo_y_scaler('random_forest_modelo_aif.pkl', 'random_forest_scaler02.pkl', 'MODELO_02.py')
modelo_hbe, scaler2 = cargar_modelo_y_scaler('random_forest_modelo_hbe.pkl', 'random_forest_scaler02.pkl', 'MODELO_02.py')
modelo3, scaler3 = cargar_modelo_y_scaler('random_forest_modelo03.pkl', 'ramdon_forest_scaler03.pkl', 'MODELO_03.py')

# Función de predicción para el primer modelo
def predecir_anemia_modelo1(input_data):
    input_data_scaled = scaler1.transform([input_data])
    prediction = modelo1.predict(input_data_scaled)[0]
    tiene_anemia = {
        0: "No tiene anemia",
        1: "Si tiene anemia"
    }
    return tiene_anemia.get(prediction)

# Función de predicción para el tercer modelo
def predecir_anemia_modelo2(input_data):
    input_data_scaled = scaler2.transform([input_data])
    
    pred_btt = modelo_btt.predict(input_data_scaled)[0]
    pred_aif = modelo_aif.predict(input_data_scaled)[0]
    pred_hbe = modelo_hbe.predict(input_data_scaled)[0]
    
    comentarios = []
    
    if pred_btt == 1:
        comentarios.append("Tiene el tipo de anemia BTT")
    if pred_aif == 1:
        comentarios.append("Tiene el tipo de anemia AIF")
    if pred_hbe == 1:
        comentarios.append("Tiene el tipo de anemia HbE")
    
    if not comentarios:
        comentarios.append("No tiene anemia")
    
    return ', '.join(comentarios)

# Función de predicción para el segundo modelo
def predecir_anemia_modelo3(input_data):
    input_data_scaled = scaler3.transform([input_data])
    prediction = modelo3.predict(input_data_scaled)[0]
    nivel_anemia = {
        1: "Grave",
        2: "Moderada",
        3: "Leve",
        4: "No tiene anemia"
    }
    return nivel_anemia.get(prediction)

@app.route('/predict/modelo1', methods=['POST'])
def predict_modelo1():
    try:
        data = request.get_json()
        print('Datos recibidos:', data)
        input_data = data['input_data']
        resultado = predecir_anemia_modelo1(input_data)
        print('Resultado de la predicción:', resultado)  # Añade esto para verificar el resultado
        return jsonify({'prediccion': resultado})
    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 500

@app.route('/predict/modelo2', methods=['POST'])
def predict_modelo2():
    data = request.get_json()
    input_data = data['input_data']
    resultado = predecir_anemia_modelo2(input_data)
    print('Resultado de la predicción modelo 2:', resultado)  # Añade esto para verificar el resultado
    return jsonify({'prediccion': resultado})

@app.route('/predict/modelo3', methods=['POST'])
def predict_modelo3():
    data = request.get_json()
    input_data = data['input_data']
    resultado = predecir_anemia_modelo3(input_data)
    print('Resultado de la predicción modelo 3:', resultado)  # Añade esto para verificar el resultado
    return jsonify({'prediccion': resultado})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
