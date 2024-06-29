import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt


data = pd.read_csv("D:\SHADOW_FOX\HousingData.csv")

from sklearn.impute import SimpleImputer

numeric_columns=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
categorical_columns=['CHAS']

imputer_numeric=SimpleImputer(strategy='median')
data[numeric_columns]=imputer_numeric.fit_transform(data[numeric_columns])

imputer_categorical=SimpleImputer(strategy='most_frequent')
data[categorical_columns]=imputer_categorical.fit_transform(data[categorical_columns])

x=data.drop('MEDV',axis=1)
y=data['MEDV']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

for i in range(10):
    print(f'predicted:{y_pred[i]:.2f},actual: {y_test.values[i]:.2f}')

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred, alpha=0.5)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs. Predicted MEDV')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  
plt.show()


def predict_medv(new_data):
    
    new_data_scaled = scaler.transform(new_data)
 
    return model.predict(new_data_scaled)


new_data = pd.DataFrame({
    'CRIM': [0.1],
    'ZN': [20],
    'INDUS': [5],
    'CHAS': [0],
    'NOX': [0.5],
    'RM': [6],
    'AGE': [65],
    'DIS': [4],
    'RAD': [1],
    'TAX': [300],
    'PTRATIO': [15],
    'B': [395],
    'LSTAT': [5]
})

predicted_medv = predict_medv(new_data)
result=predicted_medv*1000
print(f"The predicted price of the house is: {result}$") 
