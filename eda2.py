import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Lectura de la base de datos
df = pd.read_csv('DB/cars.csv', sep=';') # Cambio en el separador

'''for column in df.columns:
    print(column, df[column].dtype)'''

#Valores nulos
df_nulls = df.isnull().sum()
print(df_nulls)
df_nulls_columns = []
for column, count in df_nulls.items():
    if count != 0:
        print(f'Columna: {column} - Cantidad de nulos: {count}')
        if column == 'Averia_grave':
            df.dropna(subset=[column], inplace=True)
        elif column == 'ESTADO_CIVIL':
            df[column] = df[column].fillna('DESCONOCIDO')
        elif column == 'GENERO':
            df[column] = df[column].fillna('O')
        elif column == 'Zona _Renta':
            df[column] = df[column].fillna('Desconocido')   

print(df.isnull().sum())