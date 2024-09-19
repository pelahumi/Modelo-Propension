from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('DB/cleaned/cars_cleaned.csv', sep=',')

# One-Hot Encoding para variables categóricas
def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    return df

# Seleccionar las columnas predictoras (resto de datos)
df = df.drop(['CODE', 'EDAD_COCHE'], axis=1)
X = df.drop('Mas_1_coche', axis=1)
y = df['Mas_1_coche']

print(X.head(), '\n')

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un modelo de árbol de decisión
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

# Obtener métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar las métricas
print(f"Precisión (Accuracy): {accuracy:.2f}")
print(f"Precisión (Precision): {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Predecir el valor de Mas_1_coche para nuevos datos
new_data = pd.read_csv('DB/cars_input.csv', sep=';')
new_data = new_data.drop(['CODE', 'Revisiones'], axis=1)

new_data = label_encoder(new_data)

print(new_data.head(), '\n')

prediction = dt_model.predict(new_data)

print('No compran coche: ', np.sum(prediction == 0))
print('Compran coche: ', np.sum(prediction == 1))