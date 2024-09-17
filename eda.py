from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('DB/cleaned/cars_cleaned.csv', sep=';')

# One-Hot Encoding para variables categóricas
def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    return df

# Seleccionar las columnas predictoras (resto de datos)
df = df.drop(['CODE', 'EDAD_COCHE', 'Tiempo'], axis=1)
X = df.drop('Mas_1_coche', axis=1)
y = df['Mas_1_coche']

print(X.head(), '\n')

# Crear un modelo de árbol de decisión
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# Predecir el valor de Mas_1_coche para nuevos datos
new_data = pd.read_csv('DB/cars_input.csv', sep=';')
new_data = new_data.drop(['CODE'], axis=1)

new_data = label_encoder(new_data)

print(new_data.head(), '\n')

prediction = dt_model.predict(new_data)

print(type(prediction))

print('No compran coche: ', np.sum(prediction == 0))
print('Compran coche: ', np.sum(prediction == 1))

# Obtener métricas
accuracy = accuracy_score(y, dt_model.predict(X))
precision = precision_score(y, dt_model.predict(X), average='binary')
recall = recall_score(y, dt_model.predict(X), average='binary')
f1 = f1_score(y, dt_model.predict(X), average='binary')

# Mostrar las métricas
print(f"Precisión (Accuracy): {accuracy:.2f}")
print(f"Precisión (Precision): {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")