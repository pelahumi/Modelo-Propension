import pandas as pd

db = pd.read_csv('DB/cars.csv', sep=';')

pd.set_option('display.max_columns', None)

# Dividimos el df en los que compraron más de un coche y los que no

db_purchased = db[db['Mas_1_coche'] == 1]
db_not_purchased = db[db['Mas_1_coche'] == 0]

# Eliminamos la columna Mas_1_coche

db_purchased.drop(['Mas_1_coche'], axis=1)
db_not_purchased.drop(['Mas_1_coche'], axis=1)


# Vemos los datos que tenemos

#print(db_purchased.groupby('Tiempo')['CODE'].count())
#print(db_not_purchased['CODE'].count())

df_purchased = db_purchased.groupby('Tiempo').mean()
df_not_purchased = db_not_purchased.groupby('Tiempo').mean()


# Datos sobre Edad y Descuentos

print(df_purchased.mean())
print(df_not_purchased)

print('\n')

print('#'*100)

print('\n')

obj_cols = db.select_dtypes(include='object').columns

for col in obj_cols:
    df_purchased_col = db_purchased.groupby(col)['CODE'].count()
    df_not_purchased_col = db_not_purchased.groupby(col)['CODE'].count()
    print(df_purchased_col)
    print('\n')
    print(df_not_purchased_col)
    print('\n')
    print('#'*10)
    print('\n')
