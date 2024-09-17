import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Lectura de la base de datos
db = pd.read_csv('DB/cars.csv', sep=';')

#Cambiar tipo de datos
db.drop(['CODE'], axis=1, inplace=True)

cat_cols = db.select_dtypes(include='object').columns
cat_map = {}

le = LabelEncoder()
for col in cat_cols:
    db[col] = le.fit_transform(db[col])
    cat_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

#Ver el diccionario de las categorias
for col, mapping in cat_map.items():
    print(f"Columna: {col}")
    for category, encoded_value in mapping.items():
        print(f"  {category}: {encoded_value}")


