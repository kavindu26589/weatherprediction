# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
file_path = '/kaggle/input/fullweatherdata/colombo_weather for 50 years.csv'  # Replace with your file path
data = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

# Display the first few rows
print(data.head())

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Dataset summary
print("Dataset Info:")
print(data.info())

# Fill missing values with the column mean for numerical columns
data = data.fillna(data.mean())

# Verify there are no missing values
print("Missing Values After Cleaning:\n", data.isnull().sum())

# Extract month and day from the datetime index
data['month'] = data.index.month
data['day'] = data.index.day

# Add a 'weather' category based on precipitation
def classify_weather(row):
    if row['prcp'] == 0:
        return 'clear'
    elif row['prcp'] < 5:
        return 'cloudy'
    else:
        return 'rainy'

# Apply the function to create a weather category
data['weather'] = data.apply(classify_weather, axis=1)

# Encode the weather category
label_encoder = LabelEncoder()
data['weather_encoded'] = label_encoder.fit_transform(data['weather'])

# Check the unique weather categories
print("Weather Categories:", label_encoder.classes_)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import pickle

# Define models for each numerical feature
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
regressors = {}

# Hyperparameter grid for RandomForestRegressor
param_grid_regressor = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Train regression models for each numerical feature
for feature in features:
    # Prepare the data
    X = data[['month', 'day']]  # Use 'month' and 'day' as input features
    y = data[feature]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and tune Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid_regressor, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Save the best model
    best_regressor = grid_search.best_estimator_
    with open(f'{feature}_model.pkl', 'wb') as file:
        pickle.dump(best_regressor, file)
    
    # Evaluate the tuned model
    y_pred = best_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Feature: {feature}, Best Parameters: {grid_search.best_params_}, RMSE: {rmse}")
    
    # Store the model
    regressors[feature] = best_regressor

# Prepare the data for weather classification
X = data[['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']]  # Use predicted features
y = data['weather_encoded']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid for RandomForestClassifier
param_grid_classifier = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

# Initialize and tune Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
grid_search_classifier = GridSearchCV(estimator=classifier, param_grid=param_grid_classifier, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search_classifier.fit(X_train, y_train)

# Save the best model
best_classifier = grid_search_classifier.best_estimator_

# Save the weather classification model
with open('weather_model.pkl', 'wb') as file:
    pickle.dump(best_classifier, file)

# Save the LabelEncoder (ensure this was defined earlier during preprocessing)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Evaluate the tuned model
y_pred = best_classifier.predict(X_test)
print("Best Parameters for Weather Classification Model:", grid_search_classifier.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Input date for prediction
input_date = pd.Timestamp('2024-11-18')  # Replace with the desired date
month = input_date.month
day = input_date.day

# Predict numerical features
predicted_values = {}
for feature in features:
    # Load the model
    with open(f'{feature}_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Predict the feature
    predicted_values[feature] = model.predict([[month, day]])[0]

# Display predicted numerical features
print("Predicted Values:")
for key, value in predicted_values.items():
    print(f"{key}: {value}")

# Predict weather category
weather_input = pd.DataFrame([predicted_values])  # Create input DataFrame for the classifier
with open('weather_model.pkl', 'rb') as file:
    weather_classifier = pickle.load(file)

predicted_weather = label_encoder.inverse_transform(weather_classifier.predict(weather_input))
print("Predicted Weather:", predicted_weather[0])

# Define features to predict
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']

# Load all models
regressors = {}
for feature in features:
    with open(f'{feature}_model.pkl', 'rb') as file:
        regressors[feature] = pickle.load(file)

# Load the weather classification model
with open('weather_model.pkl', 'rb') as file:
    weather_classifier = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Predict for the next week
next_week_predictions = []

# Start from today
today = datetime.now()
for i in range(7):  # Loop for the next 7 days
    prediction_date = today + timedelta(days=i)
    month = prediction_date.month
    day = prediction_date.day
    
    # Predict numerical features for the day
    predicted_values = {}
    for feature in features:
        # Create input DataFrame with proper feature names
        input_features = pd.DataFrame([[month, day]], columns=['month', 'day'])
        predicted_values[feature] = regressors[feature].predict(input_features)[0]
    
    # Add the predicted numerical features to a DataFrame for classification
    weather_input = pd.DataFrame([predicted_values])
    
    # Predict weather category
    predicted_weather = weather_classifier.predict(weather_input)
    predicted_weather_decoded = label_encoder.inverse_transform(predicted_weather)[0]
    
    # Append the results for the day
    next_week_predictions.append({
        'date': prediction_date.strftime('%Y-%m-%d'),
        'tavg': predicted_values['tavg'],
        'tmin': predicted_values['tmin'],
        'tmax': predicted_values['tmax'],
        'prcp': predicted_values['prcp'],
        'wspd': predicted_values['wspd'],
        'pres': predicted_values['pres'],
        'weather': predicted_weather_decoded
    })

# Convert to DataFrame for easy visualization
next_week_df = pd.DataFrame(next_week_predictions)

# Display predictions for the next week
print(next_week_df)
