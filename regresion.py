import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
db = pd.read_csv('DB/cars.csv', sep=';')

db_filtered = db[db['Mas_1_coche'] == 1]

db_filtered.drop(['Mas_1_coche'], axis=1, inplace=True)

num_col = db_filtered.select_dtypes(include=['float64', 'int64']).columns







