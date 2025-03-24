from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.neighbors import KernelDensity

app = Flask(__name__)

# 📌 Cargar el modelo entrenado
model_path = 'model/earthquake_model.joblib'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ El archivo {model_path} no existe.")

model = joblib.load(model_path)

# 📌 Cargar datos históricos de terremotos
data_path = 'earthquakes.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"⚠️ El archivo {data_path} no existe.")

try:
    df_historical = pd.read_csv(data_path)

    # ✅ Verificar que tiene columnas necesarias
    if not {'latitude', 'longitude'}.issubset(df_historical.columns):
        raise ValueError("⚠️ El archivo CSV debe contener las columnas 'latitude' y 'longitude'.")

    # ✅ Eliminar filas con valores nulos en las coordenadas
    df_historical = df_historical.dropna(subset=['latitude', 'longitude'])

except Exception as e:
    raise ValueError(f"⚠️ Error al cargar {data_path}: {e}")

# 📌 Función para calcular la densidad sísmica
def calcular_densidad_sismica(latitude, longitude, df, radio=1.0):
    if df.empty:
        return 0  # Si no hay datos, devuelve 0 en la densidad

    # 📌 Convertir a radianes si los datos están en grados
    coords = np.radians(df[['latitude', 'longitude']].values)
    kde = KernelDensity(kernel='gaussian', bandwidth=radio)
    kde.fit(coords)

    # 📌 Calcular la densidad en la ubicación dada
    sample_coords = np.radians([[latitude, longitude]])
    log_densidad = kde.score_samples(sample_coords)
    return np.exp(log_densidad)[0]

# 📌 Ruta principal (interfaz web)
@app.route('/')
def home():
    return render_template('index.html')

# 📌 Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos.'}), 400

        # 📌 Obtener y validar datos del usuario
        try:
            mag = float(data['mag'])
            depth = float(data['depth'])
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
        except (ValueError, KeyError):
            return jsonify({'error': '⚠️ Faltan valores o son inválidos.'}), 400

        # 📌 Calcular la densidad sísmica automáticamente
        densidad_sismica = calcular_densidad_sismica(latitude, longitude, df_historical)

        # 📌 Crear un array con los datos para la predicción
        features = np.array([[latitude, longitude, mag, depth, densidad_sismica]])

        # 📌 Verificar que `features` tenga el mismo formato que el modelo espera
        if features.shape[1] != model.n_features_in_:
            return jsonify({'error': f"⚠️ El modelo espera {model.n_features_in_} características, pero recibió {features.shape[1]}."}), 400

        # 📌 Obtener la probabilidad de terremoto
        probability = model.predict_proba(features)[0][1]  # Probabilidad de clase 1 (terremoto)
        prediction = int(probability >= 0.3)  # Predicción binaria
        
        print("✅ Predicción:", prediction, "Probabilidad:", probability)

        return jsonify({
            'prediction': prediction,
            'probability': float(probability),
            'densidad_sismica': float(densidad_sismica)
        })

    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return jsonify({'error': str(e)}), 500

# 📌 Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
