import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Lectura de la base de datos
db = pd.read_csv('DB/cars.csv')

#Valores nulos
db.isnull().sum()

#Columnas de la base de datos
db.columns

#Valors duplicados
db.duplicated().sum()

#Tipo de datos por columna
print(db.dtypes)


