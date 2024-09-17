from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv('DB/cars.csv', sep=';')

# One-Hot Encoding para variables categóricas
def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    return df

df = label_encoder(df)

# Seleccionar las columnas predictoras (resto de datos)
df = df.drop(['CODE', 'EDAD_COCHE', 'Tiempo'], axis=1)
X = df.drop('Mas_1_coche', axis=1)

print(X.head(), '\n')

# Crear un modelo de árbol de decisión
dt_model = DecisionTreeRegressor()
dt_model.fit(X, df['Mas_1_coche'])

# Predecir el valor de Mas_1_coche para nuevos datos
new_data = pd.read_csv('DB/cars_input.csv', sep=';')
new_data = new_data.drop(['CODE'], axis=1)

new_data = label_encoder(new_data)

print(new_data.head(), '\n')

prediction = dt_model.predict(new_data)

print(type(prediction))

print('No compran coche: ', np.sum(prediction == 0))
print('Compran coche: ', np.sum(prediction == 1))