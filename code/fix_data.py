import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, roc_auc_score, 
                             precision_recall_curve, make_scorer)

from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     StratifiedKFold, learning_curve)





def label_encoder(df):
    cat_col = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    label_mappings = {}
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
        # Guardar el mapeo original de valores
        label_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))
    return df, label_mappings

def previous_analysis():
    #Lectura de la base de datos
    df = pd.read_csv('DB/cars.csv', sep=';') # Cambio en el separador
    df.head()
    df_nulls = pd.DataFrame(df.isnull().sum(), index=df.columns, columns=['Nulos'])
    null_df = df[['Averia_grave', 'ESTADO_CIVIL', 'GENERO', 'Zona _Renta']]

    null_cols = []
    # Sustituimos valores nulos
    for column, count in df_nulls['Nulos'].items():
        if count != 0:
            null_cols.append(column)
            if column == 'Averia_grave':
                df.dropna(subset=[column], inplace=True)
            elif column == 'ESTADO_CIVIL':
                df[column] = df[column].fillna('DESCONOCIDO')
            elif column == 'GENERO':
                df[column] = df[column].fillna('O')
            elif column == 'Zona _Renta':
                df[column] = df[column].fillna('Desconocido')   

    # Una vez que se han eliminado los valores nulos, verificamos que hemos hecho la transoformacion correctamente
    df_nulls = pd.DataFrame(df.isnull().sum(), index=df.columns, columns=['Nulos'])
    df_nulls.loc[null_cols]

    df.drop(['Tiempo', 'Revisiones'], axis=1, inplace=True)

    df_outliers = df[['COSTE_VENTA', 'km_anno']]
    Q1 = df_outliers.quantile(0.25)
    Q3 = df_outliers.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_outliers[(df_outliers < lower_bound) | (df_outliers > upper_bound)]
    # Eliminar filas con outliers en 'COSTE_VENTA' y 'km_anno'
    db_cleaned = df[~((df['COSTE_VENTA'].isin(outliers['COSTE_VENTA'])) | (df['km_anno'].isin(outliers['km_anno'])))]
    #df.to_csv('DB/cleaned/cars_cleaned_outliers.csv', sep=',', index=False)

    df, label_mappings = label_encoder(df)

    if not os.path.exists('DB/cleaned'):
        os.mkdir('DB/cleaned')
    df.to_csv('DB/cleaned/cars_cleaned.csv', index=False)

    df = pd.read_csv('DB/cleaned/cars_cleaned.csv')

    return df, label_mappings






def predict_model(df):
    '''# Preparar los datos
    if col_drop in ['CODE', 'EDAD_COCHE', 'Mas_1_coche']:
        return None'''
    
    df = df.drop(['CODE', 'EDAD_COCHE'], axis=1)
    X = df.drop('Mas_1_coche', axis=1)
    y = df['Mas_1_coche']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # model = GradientBoostingClassifier(learning_rate=0.01, max_depth=10, n_estimators=120)
    #model = RandomForestClassifier(max_depth=20, min_samples_leaf=2, n_estimators=160)

    # model = GradientBoostingClassifier()
    model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=6, min_samples_split=5, min_samples_leaf=3)

    # Entrenar el modelo con la combinación de hiperparámetros actual
    model.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Evaluar y mostrar los resultados
    evaluate_model(model, y_test, y_pred)

    return model, X, y, X_test, y_test, y_pred





# Creamos una funcion para evaluar las predicciones del modelo, ya que vamos a reutilizar este codigo mucho
def evaluate_model(model, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precisión (Accuracy): {round(accuracy, 2)}")
    print(f"Precisión (Precision): {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(f1, 2)}")

    '''# Predecir el valor de Mas_1_coche para nuevos datos
    new_data = pd.read_csv('DB/cars_input.csv', sep=';')
    new_data = new_data.drop(['CODE', 'Revisiones'], axis=1)
    new_data = label_encoder(new_data)
    prediction = model.predict(new_data)

    print('No compran coche: ', round(np.sum(prediction == 0)/len(prediction), 2), '%')
    print('Compran coche: ', round(np.sum(prediction == 1)/len(prediction), 2), '%')'''



def make_confusion_matrix(y_test, y_pred):
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    #plt.savefig('Img/confusion_matrix.png')
    plt.show()



def make_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'AUC: {auc:.2f}')

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend([f'ROC Curve (AUC = {auc:.2f})'])
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

    return y_pred_proba



def make_precision_recall_f1(y_test, y_pred_proba):
    # Curvas de precisión y recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Hallamos el f1 score para cada umbral
    f1_scores = 2 * (precision * recall) / (precision + recall)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mostrar las curvas de precision y recall
    axes[0].plot(recall, precision, marker='.', label='Curva Precision-Recall')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Curva Precision-Recall')
    axes[0].legend()
    axes[0].grid(True)

    # Mostramos el f1
    axes[1].plot(thresholds, f1_scores[:-1], marker='.', color='orange', label='F1-Score')
    axes[1].set_xlabel('Umbrales')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score en función del umbral')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    #plt.savefig('Img/precision_recall_f1.png')
    plt.show()



def make_cross_validation(model, X, y):
    # Definimos la validación cruzada con 5 folds
    kf = StratifiedKFold(n_splits=5)

    # Evaluación de la precisión
    precision_scorer = make_scorer(precision_score)
    precision_scores = cross_val_score(model, X, y, cv=kf, scoring=precision_scorer)

    # Mostramos los resultados de precisión en cada fold y el promedio
    print("Precisión en cada fold:", precision_scores)
    print("Precisión promedio:", precision_scores.mean())
    print('\n')

    # Mostramos la grafica
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(precision_scores)+1), precision_scores, marker='o', linestyle='--', color='blue', label='Precisión por Fold')
    plt.axhline(y=precision_scores.mean(), color='red', linestyle='-', label=f'Precisión promedio: {precision_scores.mean():.3f}')
    plt.title('Validación cruzada - Precisión en cada Fold')
    plt.xlabel('Fold')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    #plt.savefig('Img/cross_validation.png')
    plt.show()



def make_learning_curve(model, X, y):
    # Usamos la función learning_curve para obtener las puntuaciones en el entrenamiento y validación
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='precision', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    # Calculamos las medias y desviaciones estándar de las puntuaciones
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Mostramos la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Precisión en entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Precisión en validación")

    plt.title("Curva de aprendizaje del modelo")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    #plt.savefig('Img/learning_curve.png')
    plt.show()



def zona_renta_groupby(df):
    # Agrupar el DataFrame por la columna 'Zona _Renta'
    zona_renta_groups = df.groupby('Zona _Renta')
    
    # Generador que devuelve cada DataFrame del grupo
    for zona_renta, data in zona_renta_groups:
        yield zona_renta, data



def decode_zona_renta(encoded_value, label_mappings, column='Zona _Renta'):
    inverse_mapping = {v: k for k, v in label_mappings[column].items()}
    return inverse_mapping.get(encoded_value, "Valor no encontrado")



if __name__ == '__main__':

    df, label_mappings = previous_analysis()
    grouped_dfs = zona_renta_groupby(df)
    for zona_renta_encoded, df_renta in grouped_dfs:
        zona_renta_original = decode_zona_renta(zona_renta_encoded, label_mappings, column='Zona _Renta')
        print('\n', zona_renta_original)
        print('Edad cliente media: ', df_renta['Edad Cliente'].mean())
        print('Renovacion media: ', df_renta['Mas_1_coche'].mean())
        print('Coste venta media: ', df_renta['COSTE_VENTA'].mean())
        predict_model(df)
    
    '''model, X, y, X_test, y_test, y_pred = predict_model(df)
    make_confusion_matrix(y_test, y_pred)
    y_pred_proba = make_roc_curve(model, X_test, y_test)
    make_precision_recall_f1(y_test, y_pred_proba)
    make_cross_validation(model, X, y)
    make_learning_curve(model, X, y)'''