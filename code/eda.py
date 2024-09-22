from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv('DB/cleaned/cars_cleaned.csv', sep=',')

# Diccionario de hiperparámetros para cada modelo
param_grids = {
    'DecisionTreeClassifier': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'GradientBoostingClassifier': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
}

# One-Hot Encoding para variables categóricas
def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    return df

def predict_model(df, model: str = 'DecisionTreeClassifier'):
    # Seleccionar las columnas predictoras (resto de datos)
    df = df.drop(['CODE', 'EDAD_COCHE'], axis=1)
    X = df.drop('Mas_1_coche', axis=1)
    y = df['Mas_1_coche']

    #print(X.head(), '\n')

    # Separar los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear un modelo de árbol de decisión
    if model == 'DecisionTreeClassifier':
        base_model = DecisionTreeClassifier()
    elif model == 'KNeighborsClassifier':
        base_model = KNeighborsClassifier()
    elif model == 'RandomForestClassifier':
        base_model = RandomForestClassifier()
    elif model == 'GradientBoostingClassifier':
        base_model = GradientBoostingClassifier()
    elif model == 'AdaBoostClassifier':
        base_model = AdaBoostClassifier()
    else:
        raise ValueError('Modelo no reconocido')
    
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grids[model], cv=5, scoring='precision', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo optimizado
    best_model = grid_search.best_estimator_
    print(f"\nMejores hiperparámetros para {model}: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)

    # Obtener métricas
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

    # Predecir el valor de Mas_1_coche para nuevos datos
    new_data = pd.read_csv('DB/cars_input.csv', sep=';')
    new_data = new_data.drop(['CODE', 'Revisiones'], axis=1)

    new_data = label_encoder(new_data)

    prediction = best_model.predict(new_data)

    print('No compran coche: ', np.sum(prediction == 0))
    print('Compran coche: ', np.sum(prediction == 1))



models = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier']
for model in models:
    predict_model(df, model=model)