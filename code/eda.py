from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np



# One-Hot Encoding para variables categóricas
def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    return df

def predict_model(df):
    df = df.drop(['CODE', 'EDAD_COCHE'], axis=1)
    X = df.drop('Mas_1_coche', axis=1)
    y = df['Mas_1_coche']

    #print(X.head(), '\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier()    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Mostrar las métricas
    print('-------------- ', model ,' --------------')
    print(f"Precisión (Accuracy): {accuracy:.2f}")
    print(f"Precisión (Precision): {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    '''# Predecir el valor de Mas_1_coche para nuevos datos
    new_data = pd.read_csv('DB/cars_input.csv', sep=';')
    new_data = new_data.drop(['CODE', 'Revisiones'], axis=1)

    new_data = label_encoder(new_data)

    prediction = model.predict(new_data)

    print('No compran coche: ', np.sum(prediction == 0))
    print('Compran coche: ', np.sum(prediction == 1))'''



if __name__ == '__main__':
    df = pd.read_csv('DB/cars.csv', sep=';')
    df = label_encoder(df)
    predict_model(df)

    '''data_filtered = df[df['Mas_1_coche'] == 1]
    print(data_filtered.head())'''