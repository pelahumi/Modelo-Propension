import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Lectura de la base de datos
db = pd.read_csv('DB/cars.csv', sep=';')


#Graficar correlacion
def heatmap():
    num_col = db.select_dtypes(include = ['float64', 'int64']).columns
    corr = db[num_col].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Heatmap de correlacion')
    plt.show()

#Graficar boxplot

def boxplot():
    num_col = db.select_dtypes(include = ['float64', 'int64']).columns
    for col in num_col:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=db[col])
        plt.title(f'Boxplot de {col}')
        plt.show()


#Eliminamos las columnas con mayor correlacion
db.drop(['Tiempo', 'Revisiones'], axis=1, inplace=True)

#Cambiar tipo de datos
db.drop(['CODE'], axis=1, inplace=True)

cat_cols = db.select_dtypes(include='object').columns
cat_map = {}

le = LabelEncoder()
for col in cat_cols:
    db[col] = le.fit_transform(db[col])
    cat_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

#Ver el diccionario de las categorias
"""for col, mapping in cat_map.items():
    print(f"Columna: {col}")
    for category, encoded_value in mapping.items():
        print(f"  {category}: {encoded_value}")"""

if __name__ == '__main__':
    boxplot()



