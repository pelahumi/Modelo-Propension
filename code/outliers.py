import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





'''VALORES NULOS'''
# Cargar el DataFrame
db = pd.read_csv('DB/cars.csv', sep=';') # Cambio en el separador

df_nulls = pd.DataFrame(db.isnull().sum(), index=db.columns, columns=['Nulos'])

null_cols = []
# Sustituimos valores nulos
for column, count in df_nulls['Nulos'].items():
    if count != 0:
        null_cols.append(column)
        if column == 'Averia_grave':
            db.dropna(subset=[column], inplace=True)
        elif column == 'ESTADO_CIVIL':
            db[column] = db[column].fillna('DESCONOCIDO')
        elif column == 'GENERO':
            db[column] = db[column].fillna('O')
        elif column == 'Zona _Renta':
            db[column] = db[column].fillna('Desconocido')   

# Una vez que se han eliminado los valores nulos, verificamos que hemos hecho la transoformacion correctamente
df_nulls = pd.DataFrame(db.isnull().sum(), index=db.columns, columns=['Nulos'])
df_nulls.loc[null_cols]





'''HEATMAP Y BOXPLOTS'''
db.drop(['Tiempo', 'Revisiones'], axis=1, inplace=True)





'''OUTLIERS'''
# Seleccionar las columnas de interés
db_out = db[['COSTE_VENTA', 'km_anno']]
print("\n\nConteo inicial:\n", db_out.count(), '\n')

# Definir cuartiles solo para las columnas 'COSTE_VENTA' y 'km_anno'
Q1 = db_out.quantile(0.25)
Q3 = db_out.quantile(0.75)
ICR = Q3 - Q1  # Intervalo intercuartil

# Definir los límites inferior y superior
lower_bound = Q1 - 1.5 * ICR
upper_bound = Q3 + 1.5 * ICR

# Identificar los outliers
outliers = db_out[(db_out < lower_bound) | (db_out > upper_bound)]
print('Conteo de outliers: ', outliers.count())

# Eliminar filas con outliers en 'COSTE_VENTA' y 'km_anno'
db_cleaned = db[~((db['COSTE_VENTA'].isin(outliers['COSTE_VENTA'])) | (db['km_anno'].isin(outliers['km_anno'])))]

print("\n\nConteo final:\n", db_cleaned[['COSTE_VENTA', 'km_anno']].count(), '\n')

# Boxplots para verificar visualmente los outliers
'''for col in db_out.columns:
    sns.boxplot(db_cleaned[col])
    plt.title(f'Boxplot de {col}')
    plt.show()'''
