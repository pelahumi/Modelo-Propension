from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('DB/cars.csv', sep=';')

print(df.head(), '\n')

# Seleccionar las columnas predictoras (resto de datos)
df = df.drop(['CODE'], axis=1)
X = df.drop('Mas_1_coche', axis=1)

# Obtenemos las variables categoricas y numericas
categorical_vars = []
num_vars = []
for column, type in df.dtypes.items():  
    if type == 'object':
        categorical_vars.append(column)
    else:
        num_vars.append(column)

for column in df.columns:
    if column in categorical_vars:
        print('categorical: ', column)
    elif column in num_vars:
        print('numerical: ', column)
    else:
        print('error: ', column)

print('\n')

# One-Hot Encoding para variables categóricas
X = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

print(X.head(), '\n')

# Crear un modelo de regresión lineal
lr_model = LinearRegression()
lr_model.fit(X, df['Mas_1_coche'])

# Predecir el valor de Mas_1_coche para nuevos datos
new_data = pd.read_csv('DB/cars_input.csv', sep=';')

print(new_data.head(), '\n')

prediction = lr_model.predict(new_data)
