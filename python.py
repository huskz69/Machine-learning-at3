import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#phase 1

df_raw = pd.read_csv('master_markbook.csv', encoding="latin-1")
df_clean = df_raw.dropna()
df_clean = df_clean [
    df_clean['Maths_Advanced'].between(0,100) &
    df_clean['Physics'].between(0,100) &
    df_clean['Software_Engineering_Final'].between(0,100)
]

#phase 2

X1, y = df_clean[['Maths_Advanced']], df_clean['Software_Engineering_Final']
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=69)
model = LinearRegression()
model.fit(X1_train, y_train)
prediction = model.predict(X1_test)

rmse = np.sqrt(mean_squared_error(y_test, prediction))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#phase 3
weight = model.coef_[0]
bias = model.intercept_

print(f"Weight: {weight}")
print(f"Bias: {bias}")

plt.scatter(X1_test,y_test, color='blue')
 
plt.xlabel('X1 Test Data')
plt.ylabel('Y Test Data')
plt.title('X1_test vs y_test')

plt.show()