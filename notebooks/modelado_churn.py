# Modelado Predictivo de Churn en Telecomunicaciones
# =============================================

# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report, 
                            precision_recall_curve, roc_curve, auc)
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import plot_importance
try:
    import shap
except ImportError:
    print("Biblioteca SHAP no disponible. Se omitirá la interpretación con SHAP.")
import warnings
warnings.filterwarnings('ignore')

# Configuraciones para visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)

# Cargar datos preprocesados
def load_prepared_data(file_path):
    """
    Carga los datos ya preprocesados para el modelado,
    los conjuntos X_train, X_test, y_train, y_test.
    """
    # Leer el CSV con punto y coma como separador
    df = pd.read_csv(file_path, sep=';')
    
    # Formatear columnas numéricas que usan coma como separador decimal
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass
    
    # Codificar variables categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'Customer_ID' in categorical_columns:
        categorical_columns.remove('Customer_ID')
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Separar características y variable objetivo
    X = df_encoded.drop(['churn', 'Customer_ID'], axis=1, errors='ignore')
    y = df_encoded['churn']
    
    # Escalar características
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Datos cargados: {X_train.shape[0]} filas de entrenamiento, {X_test.shape[0]} filas de prueba")
    return X_train, X_test, y_train, y_test, X.columns

# 1. Entrenamiento de modelo base (Regresión Logística)
def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n=== MODELO DE REGRESIÓN LOGÍSTICA ===")
    
    # Entrenar modelo base
    logreg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    logreg.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    
    # Evaluar modelo
    print("\nResultados de Regresión Logística:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión - Regresión Logística')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.show()
    
    # Curva ROC
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Regresión Logística')
    plt.legend()
    plt.show()
    
    # Importancia de características
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': logreg.coef_[0]
    })
    
    # Ordenar por valor absoluto para ver las más importantes (sin importar la dirección)
    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top_features = coefficients.head(15)
    sns.barplot(x='Coefficient', y='Feature', data=top_features)
    plt.title('Top 15 Características Más Influyentes - Regresión Logística')
    plt.tight_layout()
    plt.show()
    
    return logreg, y_pred_proba

# 2. Entrenamiento de Random Forest
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n=== MODELO RANDOM FOREST ===")
    
    # Definir el modelo base
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Entrenar con validación cruzada para ver rendimiento
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"AUC-ROC con validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Optimizar hiperparámetros con búsqueda en cuadrícula
    print("\nOptimizando hiperparámetros...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor AUC-ROC en validación: {grid_search.best_score_:.4f}")
    
    # Obtener el mejor modelo
    best_rf = grid_search.best_estimator_
    
    # Hacer predicciones
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Evaluar modelo
    print("\nResultados de Random Forest optimizado:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión - Random Forest')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.show()
    
    # Curva ROC
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Random Forest')
    plt.legend()
    plt.show()
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Características Más Importantes - Random Forest')
    plt.tight_layout()
    plt.show()
    
    # Visualizar un árbol del bosque (limitado a profundidad 3 para legibilidad)
    plt.figure(figsize=(20, 10))
    plot_tree(best_rf.estimators_[0], max_depth=3, feature_names=X_train.columns, 
              filled=True, rounded=True, class_names=['No Churn', 'Churn'])
    plt.title('Visualización de un Árbol de Decisión del Random Forest (limitado a profundidad 3)')
    plt.show()
    
    return best_rf, y_pred_proba

# 3. Entrenamiento de XGBoost
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n=== MODELO XGBOOST ===")
    
    # Definir el modelo base
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Equilibrar clases
    )
    
    # Entrenar con validación cruzada para ver rendimiento inicial
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"AUC-ROC con validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Optimizar hiperparámetros con búsqueda aleatoria (más eficiente que grid search para XGBoost)
    print("\nOptimizando hiperparámetros...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=10,  # Número de combinaciones aleatorias a probar
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"Mejor AUC-ROC en validación: {random_search.best_score_:.4f}")
    
    # Obtener el mejor modelo
    best_xgb = random_search.best_estimator_
    
    # Hacer predicciones
    y_pred = best_xgb.predict(X_test)
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    # Evaluar modelo
    print("\nResultados de XGBoost optimizado:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión - XGBoost')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.show()
    
    # Curva ROC
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - XGBoost')
    plt.legend()
    plt.show()
    
    # Importancia de características
    plt.figure(figsize=(12, 10))
    plot_importance(best_xgb, max_num_features=15, height=0.8, title='Importancia de Características - XGBoost')
    plt.show()
    
    # Cálculo de importancia de características basada en ganancia
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_xgb.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Características Más Importantes - XGBoost')
    plt.tight_layout()
    plt.show()
    
    return best_xgb, y_pred_proba

# 4. Comparación e interpretación de modelos
def compare_models(models_dict, X_test, y_test):
    print("\n=== COMPARACIÓN DE MODELOS ===")
    
    # Comparar métricas de rendimiento
    results = {}
    
    for name, (model, y_pred_proba) in models_dict.items():
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-ROC': auc_roc
        }
    
    # Convertir a DataFrame para visualización
    results_df = pd.DataFrame(results).T
    
    print("\nComparación de métricas:")
    print(results_df)
    
    # Visualizar comparación de métricas
    plt.figure(figsize=(14, 10))
    results_df.plot(kind='bar', figsize=(14, 10))
    plt.title('Comparación de Métricas de Rendimiento por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Valor')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(title='Métrica')
    plt.tight_layout()
    plt.show()
    
    # Comparar curvas ROC
    plt.figure(figsize=(10, 8))
    
    for name, (model, y_pred_proba) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Comparación de Curvas ROC')
    plt.legend(loc='lower right')
    plt.show()
    
    # Identificar el mejor modelo basado en AUC-ROC
    best_model_name = results_df['AUC-ROC'].idxmax()
    print(f"\nMejor modelo según AUC-ROC: {best_model_name} ({results_df.loc[best_model_name, 'AUC-ROC']:.4f})")
    
    return results_df, best_model_name

# 5. Interpretación avanzada del mejor modelo con SHAP
def interpret_with_shap(best_model, X_train, X_test, feature_names):
    try:
        print("\n=== INTERPRETACIÓN CON SHAP ===")
        
        # Crear explainer SHAP
        if isinstance(best_model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.TreeExplainer(best_model)
        
        # Calcular valores SHAP (usar una muestra para eficiencia)
        sample_size = min(100, X_test.shape[0])
        X_sample = X_test.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Para modelos no-XGBoost, el formato puede ser diferente
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Tomar los valores para la clase positiva
        
        # Gráfico resumen de valores SHAP
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('Resumen de Valores SHAP - Impacto de Características en la Predicción')
        plt.tight_layout()
        plt.show()
        
        # Gráfico de dependencia para las características más importantes
        top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]  # Top 5 características
        
        for feature_idx in top_features:
            feature_name = feature_names[feature_idx]
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(feature_idx, shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f'Gráfico de Dependencia SHAP para {feature_name}')
            plt.tight_layout()
            plt.show()
        
        # Waterfall plot para un ejemplo específico
        sample_idx = 0  # Primer ejemplo del conjunto de prueba
        plt.figure(figsize=(12, 8))
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[sample_idx], 
                                              features=X_sample.iloc[sample_idx], feature_names=feature_names, show=False)
        plt.title('Waterfall Plot - Contribución de Características para un Cliente Específico')
        plt.tight_layout()
        plt.show()
        
        return explainer, shap_values
    
    except Exception as e:
        print(f"Error al usar SHAP: {e}")
        print("Omitiendo análisis SHAP. Asegúrese de tener instalada la biblioteca SHAP correctamente.")
        return None, None

# 6. Función para predecir y explicar el churn para un cliente específico
def predict_and_explain_customer(model, explainer, customer_data, feature_names):
    print("\n=== PREDICCIÓN Y EXPLICACIÓN PARA UN CLIENTE ESPECÍFICO ===")
    
    # Asegurarse de que los datos del cliente tengan la misma estructura que los datos de entrenamiento
    if customer_data.shape[1] != len(feature_names):
        print(f"Error: Los datos del cliente tienen {customer_data.shape[1]} características, pero el modelo espera {len(feature_names)}.")
        return
    
    # Realizar predicción
    churn_probability = model.predict_proba(customer_data)[0, 1]
    churn_prediction = "ABANDONARÁ" if churn_probability >= 0.5 else "NO ABANDONARÁ"
    
    print(f"Predicción para el cliente: {churn_prediction}")
    print(f"Probabilidad de abandono: {churn_probability:.4f}")
    
    # Identificar principales factores de influencia
    try:
        if explainer is not None:
            # Calcular valores SHAP
            shap_values = explainer.shap_values(customer_data)
            
            # Para modelos no-XGBoost, el formato puede ser diferente
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Tomar los valores para la clase positiva
            
            # Crear DataFrame con impacto de características
            feature_impact = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_values[0],
                'Value': customer_data.values[0]
            })
            
            # Ordenar por impacto absoluto
            feature_impact['Abs_Impact'] = np.abs(feature_impact['Impact'])
            feature_impact = feature_impact.sort_values('Abs_Impact', ascending=False)
            
            # Mostrar top 10 factores
            print("\nPrincipales factores que influyen en la predicción:")
            for i, row in feature_impact.head(10).iterrows():
                impact = "INCREMENTA" if row['Impact'] > 0 else "REDUCE"
                print(f"  - {row['Feature']}: {impact} el riesgo de abandono (impacto: {row['Impact']:.4f}, valor: {row['Value']:.2f})")
            
            # Visualizar waterfall plot
            plt.figure(figsize=(12, 8))
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], 
                                                  features=customer_data.iloc[0], feature_names=feature_names, show=False)
            plt.title('Factores que Influyen en la Predicción de Abandono del Cliente')
            plt.tight_layout()
            plt.show()
        else:
            # Alternativa si SHAP no está disponible: usar importancia de características del modelo
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_,
                    'Value': customer_data.values[0]
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                print("\nCaracterísticas más importantes del modelo (sin interpretación específica para este cliente):")
                for i, row in feature_importance.head(10).iterrows():
                    print(f"  - {row['Feature']}: importancia {row['Importance']:.4f}, valor cliente: {row['Value']:.2f}")
            elif hasattr(model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_[0],
                    'Value': customer_data.values[0]
                })
                feature_importance['Impact'] = feature_importance['Coefficient'] * feature_importance['Value']
                feature_importance = feature_importance.sort_values('Impact', ascending=False)
                
                print("\nFactores que contribuyen a la predicción:")
                for i, row in feature_importance.head(10).iterrows():
                    impact = "INCREMENTA" if row['Impact'] > 0 else "REDUCE"
                    print(f"  - {row['Feature']}: {impact} el riesgo de abandono (impacto: {row['Impact']:.4f}, valor: {row['Value']:.2f})")
    
    except Exception as e:
        print(f"Error al generar explicación: {e}")
    
    return churn_probability

# 7. Función principal para ejecutar todo el análisis
def main(file_path=None, X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Función principal para ejecutar todo el proceso de modelado.
    
    Args:
        file_path: Ruta al archivo CSV (si se necesita cargar los datos).
        X_train, X_test, y_train, y_test: Datos ya preparados (opcional).
    
    Returns:
        models_dict: Diccionario con los modelos entrenados.
        results_df: DataFrame con los resultados de la comparación.
        best_model_name: Nombre del mejor modelo.
    """
    # 1. Cargar datos preparados si no se proporcionan
    if X_train is None or X_test is None or y_train is None or y_test is None:
        if file_path is not None:
            X_train, X_test, y_train, y_test, feature_names = load_prepared_data(file_path)
        else:
            raise ValueError("Debe proporcionar datos preparados o la ruta al archivo CSV.")
    else:
        feature_names = X_train.columns
    
    # 2. Entrenar modelos
    print("\nEntrenando modelos...")
    logreg_model, logreg_proba = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_model, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
    
    # 3. Comparar modelos
    models_dict = {
        'Regresión Logística': (logreg_model, logreg_proba),
        'Random Forest': (rf_model, rf_proba),
        'XGBoost': (xgb_model, xgb_proba)
    }
    
    results_df, best_model_name = compare_models(models_dict, X_test, y_test)
    
    # 4. Interpretar el mejor modelo con SHAP
    best_model = models_dict[best_model_name][0]
    explainer, _ = interpret_with_shap(best_model, X_train, X_test, feature_names)
    
    # 5. Demostración: Predecir y explicar para un cliente específico
    # Tomamos un cliente del conjunto de prueba como ejemplo
    sample_customer = X_test.iloc[[0]]  # Primer cliente del conjunto de prueba
    predict_and_explain_customer(best_model, explainer, sample_customer, feature_names)
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print(f"\nMejor modelo: {best_model_name}")
    print("\nPróximos pasos recomendados:")
    print("1. Implementar el modelo en producción")
    print("2. Configurar un sistema de monitoreo para el rendimiento del modelo")
    print("3. Desarrollar estrategias de retención basadas en las variables más importantes")
    print("4. Realizar pruebas A/B para evaluar la efectividad de las estrategias de retención")
    
    return models_dict, results_df, best_model_name, best_model, explainer

# Ejecutar el análisis completo
if __name__ == "__main__":
    file_path = "dataset.csv"
    models_dict, results_df, best_model_name, best_model, explainer = main(file_path)