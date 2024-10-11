from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)



# Absolute path example
data = pd.read_csv("dataset/recipe_final.csv")


# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Combine Features
X_combined = np.hstack([X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)


# Function to Recommend Recipes
def recommend_recipes(input_ingredients):
    # Transform the input ingredients (ensure it's a string)
    input_ingredients_transformed = vectorizer.transform([input_ingredients])
    # Combine with additional features if necessary
    input_combined = np.hstack([input_ingredients_transformed.toarray()])
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_combined)
    # Get recommendations based on the nearest neighbors
    recommendations = data.iloc[indices[0]] 
    return recommendations[['recipe_id','recipe_name', 'ingredients_list', 'image_url','cooking_directions']]
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        input_features = ingredients
        recommendations = recommend_recipes(input_features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)