from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Muat model
model = joblib.load('model/best_random_forest_model_with_preprocessing.joblib')

def categorize_mpg(mpg):
    if mpg >= 25:
        return 'Irit'
    elif mpg < 15:
        return 'Boros'
    else:
        return 'Sedang'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df[['Eng Displ', 'Cylinders', '# Gears']] = df[['Eng Displ', 'Cylinders', '# Gears']].astype(float)
    prediction = model.predict(df)
    result = prediction[0]
    category = categorize_mpg(result)
    return render_template('index.html', prediction_text=f'Predicted CombMpg: {result:.2f}', category=f'Kategori: {category}')

if __name__ == '__main__':
    app.run(debug=True)