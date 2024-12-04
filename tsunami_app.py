from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de Flask
app = Flask(__name__)

# Ruta al archivo CSV (asegúrate de que esté en la misma carpeta que este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'tsunami_dataset.csv')

# Cargar datos y modelo
df = pd.read_csv(CSV_PATH)
columns_to_drop = ['DEATHS_TOTAL_DESCRIPTION', 'URL', 'HOUSES_TOTAL_DESCRIPTION',
                   'DAMAGE_TOTAL_DESCRIPTION', 'EQ_DEPTH', 'DAY', 'HOUR', 'MINUTE']
df.drop(columns=columns_to_drop, inplace=True)

def categorize_risk(intensity):
    if intensity <= 1:
        return 'Bajo'
    elif 1 < intensity <= 3:
        return 'Medio'
    else:
        return 'Alto'

df['Risk_Level'] = df['TS_INTENSITY'].apply(categorize_risk)
X = df[['YEAR', 'LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE']]
y = df['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

geolocator = Nominatim(user_agent="miAplicacionDeGeolocalizacion")

def get_country(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language="en", timeout=10)
        if location and location.raw.get('address', {}).get('country'):
            return location.raw['address']['country']
        else:
            return "Unknown"
    except:
        return "Service Timeout"

@app.route('/')
def home():
    return render_template('principal.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    predictions = clf.predict([data[:4]])
    country = get_country(data[1], data[2])
    return jsonify({'risk': predictions[0], 'country': country})

@app.route('/visualizations')
def visualizations():
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Risk_Level', palette='viridis')
    plt.title('Distribución de los Niveles de Riesgo de Tsunami')
    img_path = os.path.join('static', 'risk_distribution.png')
    plt.savefig(os.path.join(BASE_DIR, img_path))
    plt.close()
    return render_template('visualizations.html', image=img_path)
from sklearn.tree import export_graphviz
import graphviz

@app.route('/arbol.png')
def generate_tree_image():
    # Crear el archivo .dot desde el árbol de decisión
    dot_data = export_graphviz(clf.estimators_[0], 
                               out_file=None,
                               feature_names=X.columns,
                               class_names=clf.classes_,
                               filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    img_path = os.path.join(BASE_DIR, 'static', 'arbol')
    graph.format = 'png'
    graph.render(img_path, cleanup=True)  # Guarda la imagen como PNG
    
    # Comprueba si el archivo se generó
    if not os.path.exists(f"{img_path}.png"):
        return jsonify({'error': 'No se pudo generar el árbol de decisión'}), 500
    
    # Devuelve la ruta de la imagen
    return jsonify({'message': 'Árbol generado', 'image_path': '/static/arbol.png'})

if __name__ == '__main__':
    app.run(debug=True)
