import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eda import predict_model



# Cargar el DataFrame
db = pd.read_csv('DB/cleaned/cars_cleaned.csv', sep=',')

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

# Eliminar filas con outliers en 'COSTE_VENTA' y 'km_anno'
db_cleaned = db[~((db['COSTE_VENTA'].isin(outliers['COSTE_VENTA'])) | (db['km_anno'].isin(outliers['km_anno'])))]

print("\n\nConteo final:\n", db_cleaned[['COSTE_VENTA', 'km_anno']].count(), '\n')

# Boxplots para verificar visualmente los outliers
'''for col in db_out.columns:
    sns.boxplot(db_cleaned[col])
    plt.title(f'Boxplot de {col}')
    plt.show()'''

# Guardar los datos limpiados
# db_cleaned.to_csv('DB/cleaned/cars_cleaned_outliers.csv', sep=',', index=False)



# Predecir usando el modelo
predict_model(db_cleaned)
