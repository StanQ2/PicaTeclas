from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

csv_path = 'app/data.csv'

# Cargar el conjunto de datos desde el archivo CSV
videogames = pd.read_csv(csv_path)

# Limpiar los nombres de los juegos
def clean_name(name):
    if isinstance(name, str):
        return re.sub("[^a-zA-Z0-9]", " ", name)
    else:
        return name

videogames["clean_name"] = videogames["name"].apply(clean_name)

# Reemplazar los valores NaN en la columna "clean_name" con cadenas vacías
videogames['clean_name'] = videogames['clean_name'].fillna('')

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Calcular las representaciones TF-IDF
tfidf = vectorizer.fit_transform(videogames["clean_name"])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    selected_genres_input = request.form.get('genres_input')
    selected_genres = [genre.strip() for genre in selected_genres_input.split(',')]

    if 'genres' in videogames.columns:
        filtered_games = videogames[videogames['genres'].isin(selected_genres)]
        filtered_indices = filtered_games.index
        filtered_tfidf = tfidf[filtered_indices, :]

        # Convertir los resultados a un formato JSON
        results_json = filtered_games.to_json(orient='records')

        # Devolver los resultados como JSON
        return jsonify(results_json)
    else:
        return jsonify({'error': "La columna 'genres' no existe en el DataFrame."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)