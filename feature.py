import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Lectura de la base de datos
db = pd.read_csv('DB/cars.csv', sep = ';')

#Valores nulos
db.isnull().sum()

#Columnas de la base de datos
db.columns

#Valors duplicados
db.duplicated().sum()

#Tipo de datos por columna
db.dtypes

#Estadisticas de la base de datos
db.describe()

#Cambiar el tipo de dato:
db['Campanna1'] = db['Campanna1'].map({'SI' : 1, 'NO' : 0})
db['Campanna2'] = db['Campanna1'].map({'SI' : 1, 'NO' : 0})
db['Campanna3'] = db['Campanna1'].map({'SI' : 1, 'NO' : 0})


print(db['Potencia_'].unique())





