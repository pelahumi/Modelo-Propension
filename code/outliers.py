import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

db = pd.read_csv('DB/cleaned/cars_cleaned.csv', sep=',')

# Looking for outliers

# Defining quartiles
Q1 = db.quantile(0.25)
Q3 = db.quantile(0.75)
ICR = Q3 - Q1 

lower_bound = Q1 - 1.5 * ICR
upper_bound = Q3 + 1.5 * ICR

# Outliers

outliers = db[(db < lower_bound) | (db > upper_bound)].dropna(how='all')

# Drop outliers

db = db[~db.isin(outliers)].dropna(how='all')

# Boxplots

"""for col in db.columns:
    sns.boxplot(db[col])
    plt.title(f'Boxplot de {col}')
    plt.show()"""

# Save the cleanned data
db.to_csv('DB/cleaned/cars_cleaned_outliers.csv', sep=',', index=False)






