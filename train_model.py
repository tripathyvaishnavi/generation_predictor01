
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('myntradata.csv')

# Preprocess the data
df['Generation'] = df['AGE'].apply(lambda x: 'Gen Z' if 16 <= x <= 24 else ('Millennials' if 26 <= x <= 35 else 'Gen X'))

# Encode categorical variables
brand_label_encoder = LabelEncoder()
category_label_encoder = LabelEncoder()
individual_category_label_encoder = LabelEncoder()
gender_label_encoder = LabelEncoder()
generation_label_encoder = LabelEncoder()

df['BrandName'] = brand_label_encoder.fit_transform(df['BrandName'])
df['Category'] = category_label_encoder.fit_transform(df['Category'])
df['Individual_category'] = individual_category_label_encoder.fit_transform(df['Individual_category'])
df['category_by_Gender'] = gender_label_encoder.fit_transform(df['category_by_Gender'])
df['Generation'] = generation_label_encoder.fit_transform(df['Generation'])

# Define features and target variable
X = df[['BrandName', 'Category', 'Individual_category', 'category_by_Gender', 'Ratings', 'Reviews', 'OriginalPrice']]
y = df['Generation']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(model, 'generation_predictor.pkl')
joblib.dump({
    'brand': brand_label_encoder,
    'category': category_label_encoder,
    'individual_category': individual_category_label_encoder,
    'gender': gender_label_encoder,
    'generation': generation_label_encoder
}, 'label_encoders.pkl')

print("Model and label encoders saved successfully.")
