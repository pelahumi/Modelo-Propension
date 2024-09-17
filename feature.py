import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Lectura de la base de datos
db = pd.read_csv('DB/cars.csv', sep = ';')

#Diagramas de caja y heatmap

def boxplot(db):
    object_col = db.select_dtypes(include = 'object').columns
    numeric_col = db.select_dtypes(include = ['float64', 'int64']).columns

    for cat_col in object_col:
        for num_col in numeric_col:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=cat_col, y=num_col, data=db)
            plt.title(f'Boxplot de {num_col} por {cat_col}')
            plt.xticks(rotation=45)
            plt.show()

if __name__ == '__main__':
    boxplot(db)


    
