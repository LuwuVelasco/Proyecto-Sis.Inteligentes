from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import os

# Configuración de Flask
app = Flask(__name__)

# Ruta al archivo CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'tsunami_dataset.csv')

# Función para cargar y limpiar datos
def load_and_prepare_data():
    # Cargar los datos
    df = pd.read_csv(CSV_PATH)

    # Eliminar columnas innecesarias
    columns_to_drop = ['DEATHS_TOTAL_DESCRIPTION', 'URL', 'HOUSES_TOTAL_DESCRIPTION',
                       'DAMAGE_TOTAL_DESCRIPTION', 'EQ_DEPTH', 'DAY', 'HOUR', 'MINUTE']
    df.drop(columns=columns_to_drop, inplace=True)

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Crear categorías de riesgo
    def categorize_risk(intensity):
        if intensity <= 1:
            return 'Bajo'
        else:
            return 'Alto'

    df['Risk_Level'] = df['TS_INTENSITY'].apply(categorize_risk)

    # Filtrar para que solo haya "Bajo" y "Alto"
    df = df[df['Risk_Level'].isin(['Bajo', 'Alto'])]

    # Balancear las clases
    min_class_size = df['Risk_Level'].value_counts().min()
    df_balanced = df.groupby('Risk_Level').apply(lambda x: x.sample(min_class_size)).reset_index(drop=True)

    # Normalizar características numéricas
    scaler = MinMaxScaler()
    df_balanced[['YEAR', 'LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE']] = scaler.fit_transform(
        df_balanced[['YEAR', 'LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE']]
    )

    return df_balanced, scaler

# Cargar datos procesados
df, scaler = load_and_prepare_data()

# Separar características y variable objetivo
X = df[['YEAR', 'LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE']]
y = df['Risk_Level']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
clf = RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=42)
clf.fit(X_train, y_train)

# Inicializar geolocalizador
geolocator = Nominatim(user_agent="geoapi")

# Función para obtener el país usando latitud y longitud
def get_country(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language="en", timeout=10)
        if location and location.raw.get('address', {}).get('country'):
            return location.raw['address']['country']
        else:
            return "Unknown"
    except GeocoderTimedOut:
        return "Service Timeout"
    except Exception as e:
        return str(e)

# Rutas de Flask
@app.route('/')
def home():
    return render_template('principal.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del cliente
        data = request.json['data']
        year, lat, lon, mag = data

        # Normalizar los datos de entrada usando el scaler entrenado
        input_data = scaler.transform([[year, lat, lon, mag]])

        # Realizar predicción
        prediction = clf.predict(input_data)[0]

        # Obtener país
        country = get_country(lat, lon)

        return jsonify({'risk': prediction, 'country': country})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
