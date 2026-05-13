import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv('master_markbook.csv', encoding="latin-1")
df_clean = df_raw.dropna()
df_clean = df_clean [
    df_clean['Maths_Advanced'].between(0,100) &
    df_clean['Physics'].between(0,100) &
    df_clean['Software_Engineering_Final'].between(0,100)
]
print(df_clean)


