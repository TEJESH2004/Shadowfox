import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Importing the CSV data file using pandas
data = pd.read_csv("D:/SHADOW_FOX/level2/car.csv")

# Data preprocessing
data['Age'] = 2023 - data['Year']
data = data.drop(['Year', 'Car_Name'], axis=1)
numeric_columns = ['Present_Price', 'Kms_Driven', 'Age']
categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']

imputer_numeric = SimpleImputer(strategy='median')
data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])

imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])

# Encoding categorical variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Splitting data into testing and training data
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model selection
model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid, 
    n_iter=50, 
    cv=3, 
    random_state=42, 
    n_jobs=-1,
    verbose=2
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Model training and evaluation
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R² Score: {r2}')

# Saving the model
joblib.dump(best_model, 'car_price_predictor.pkl')

# Making predictions
loaded_model = joblib.load('car_price_predictor.pkl')

new_data = pd.DataFrame({
    'Present_Price': [6.0],
    'Kms_Driven': [3000],
    'Owner': [0],
    'Age': [10],
    'Fuel_Type_Diesel': [1],
    'Fuel_Type_Petrol': [0],
    'Seller_Type_Individual': [0],
    'Transmission_Manual': [0]
})

new_data_scaled = scaler.transform(new_data)

predicted_price = loaded_model.predict(new_data_scaled)
predicted_price_rupees = predicted_price[0] * 10000

print(f'Predicted Selling Price: ₹{predicted_price_rupees:,.2f}')
