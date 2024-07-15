

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('generation_predictor.pkl')
label_encoders = joblib.load('label_encoders.pkl')

brand_label_encoder = label_encoders['brand']
category_label_encoder = label_encoders['category']
individual_category_label_encoder = label_encoders['individual_category']
gender_label_encoder = label_encoders['gender']
generation_label_encoder = label_encoders['generation']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        BrandName = request.form['BrandName']
        Category = request.form['Category']
        Individual_category = request.form['Individual_category']
        category_by_Gender = request.form['category_by_Gender']
        Ratings = float(request.form['Ratings'])
        Reviews = int(request.form['Reviews'])
        OriginalPrice = float(request.form['OriginalPrice'])

        # Encode categorical variables with error handling
        try:
            BrandName_encoded = brand_label_encoder.transform([BrandName])[0]
        except ValueError:
            BrandName_encoded = -1  # Or handle as needed

        try:
            Category_encoded = category_label_encoder.transform([Category])[0]
        except ValueError:
            Category_encoded = -1  # Or handle as needed

        try:
            Individual_category_encoded = individual_category_label_encoder.transform([Individual_category])[0]
        except ValueError:
            Individual_category_encoded = -1  # Or handle as needed

        try:
            category_by_Gender_encoded = gender_label_encoder.transform([category_by_Gender])[0]
        except ValueError:
            category_by_Gender_encoded = -1  # Or handle as needed

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[BrandName_encoded, Category_encoded, Individual_category_encoded, category_by_Gender_encoded, Ratings, Reviews, OriginalPrice]],
                                  columns=['BrandName', 'Category', 'Individual_category', 'category_by_Gender', 'Ratings', 'Reviews', 'OriginalPrice'])

        # Predict the generation
        prediction = model.predict(input_data)[0]

        # Decode the prediction to get the original label
        generation = generation_label_encoder.inverse_transform([prediction])[0]

        return render_template('result.html', prediction=generation)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
