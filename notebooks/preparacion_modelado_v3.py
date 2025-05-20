# Análisis de Outliers, Selección de Características y Preparación para Modelado
# =============================================================================

# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configuraciones para visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)

# Función para formatear valores numéricos con comas como separador decimal
def format_numeric_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        # Intentar convertir columnas con formato de texto a numéricas
        try:
            # Reemplazar comas por puntos para el formato decimal
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass
    return df

# 1. Cargar los datos
def load_data(file_path):
    # Leer el CSV con punto y coma como separador
    df = pd.read_csv(file_path, sep=';')
    
    # Formatear columnas numéricas que usan coma como separador decimal
    df = format_numeric_columns(df)
    
    print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")
    return df

# 2. Análisis de Outliers
def analyze_outliers(df):
    print("\n=== ANÁLISIS DE OUTLIERS ===")
    
    # Seleccionar columnas numéricas (excluyendo la variable objetivo y el ID del cliente)
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'churn' in numeric_columns:
        numeric_columns.remove('churn')
    if 'Customer_ID' in numeric_columns:
        numeric_columns.remove('Customer_ID')
    
    # 2.1 Identificar outliers mediante el método IQR
    outliers_summary = {}
    
    for col in numeric_columns:
        # Calcular Q1, Q3 e IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contar outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = len(outliers)
        percentage = (num_outliers / len(df)) * 100
        
        # Almacenar resumen
        if percentage > 0:
            outliers_summary[col] = {
                'count': num_outliers,
                'percentage': percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    # Ordenar por porcentaje de outliers
    sorted_outliers = sorted(outliers_summary.items(), key=lambda x: x[1]['percentage'], reverse=True)
    
    # Mostrar columnas con mayor porcentaje de outliers
    print("\nColumnas con mayor porcentaje de outliers (IQR method):")
    for col, stats in sorted_outliers[:10]:  # Top 10
        print(f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%) [límites: {stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    
    # 2.2 Visualizar la distribución de las variables con más outliers
    print("\nVisualizando distribución de variables con más outliers:")
    
    for col, _ in sorted_outliers[:5]:  # Top 5 para visualización
        plt.figure(figsize=(15, 6))
        
        # Boxplot por churn
        plt.subplot(1, 2, 1)
        sns.boxplot(x='churn', y=col, data=df)
        plt.title(f'Boxplot de {col} por Churn')
        plt.xlabel('Churn')
        plt.ylabel(col)
        
        # Histograma
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x=col, hue='churn', bins=30, kde=True, alpha=0.6)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.show()
    
    # 2.3 Identificar potenciales clientes atípicos (outliers en múltiples variables)
    # Contamos cuántas variables tienen outliers para cada cliente
    outlier_counts = pd.DataFrame(index=df.index)
    
    for col, stats in outliers_summary.items():
        lower = stats['lower_bound']
        upper = stats['upper_bound']
        outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    
    outlier_counts['total_outliers'] = outlier_counts.sum(axis=1)
    
    # Clientes con más variables atípicas
    top_outlier_clients = outlier_counts.nlargest(10, 'total_outliers')
    
    print("\nClientes con mayor número de variables atípicas:")
    for idx, row in top_outlier_clients.iterrows():
        customer_id = df.loc[idx, 'Customer_ID']
        churn_status = df.loc[idx, 'churn']
        print(f"Cliente {customer_id}: {int(row['total_outliers'])} variables atípicas (Churn: {churn_status})")
    
    return outliers_summary, outlier_counts

# 3. Selección de Características
def feature_selection(df):
    print("\n=== SELECCIÓN DE CARACTERÍSTICAS ===")
    
    # Separar características y variable objetivo
    X = df.drop(['churn', 'Customer_ID'], axis=1, errors='ignore')
    y = df['churn']
    
    # 3.1 Manejar variables categóricas y valores NaN
    print("\n1. Preparación de datos para análisis de correlación:")
    
    # Identificar columnas numéricas y categóricas
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    print(f"Variables numéricas: {len(numeric_columns)}")
    print(f"Variables categóricas: {len(categorical_columns)}")
    
    # Comprobar valores NaN en columnas numéricas
    nan_counts = X[numeric_columns].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    if len(cols_with_nan) > 0:
        print("\nColumnas numéricas con valores NaN:")
        print(cols_with_nan)
        
        # Imputar valores NaN en variables numéricas con la mediana
        for col in cols_with_nan.index:
            median_value = X[col].median()
            X[col].fillna(median_value, inplace=True)
            print(f"  - {col}: {cols_with_nan[col]} valores NaN imputados con la mediana ({median_value})")
    else:
        print("\nNo hay valores NaN en las columnas numéricas.")
    
    # 3.2 Eliminar columnas con alta correlación (solo para variables numéricas)
    print("\n2. Eliminación de características numéricas altamente correlacionadas:")
    
    # Usar solo columnas numéricas para la matriz de correlación
    X_numeric = X[numeric_columns]
    
    # Calcular matriz de correlación
    correlation_matrix = X_numeric.corr().abs()
    
    # Crear una máscara triangular superior
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Encontrar columnas con correlación mayor a 0.9
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    print(f"Se identificaron {len(high_corr_cols)} columnas con alta correlación (>0.9):")
    if len(high_corr_cols) > 0:
        for col in high_corr_cols:
            corr_with = upper[col][upper[col] > 0.9].index.tolist()
            corr_values = upper[col][upper[col] > 0.9].values.tolist()
            for i, corr_col in enumerate(corr_with):
                print(f"  - {col} correlacionada con {corr_col} (r = {corr_values[i]:.3f})")
    else:
        print("  No se encontraron columnas con correlación >0.9")
    
    # Eliminar columnas con alta correlación
    X_no_corr = X.drop(columns=high_corr_cols)
    print(f"\nDimensiones después de eliminar columnas correlacionadas: {X_no_corr.shape}")
    
    # 3.3 Selección basada en ANOVA F-value para variables numéricas
    print("\n3. Selección basada en ANOVA F-value (solo variables numéricas):")
    
    # Eliminar columnas con alta correlación
    X_no_corr = X_numeric.drop(columns=high_corr_cols, errors='ignore')
    
    # Aplicar SelectKBest con f_classif (ANOVA F-value)
    k = min(20, X_no_corr.shape[1])  # Seleccionar hasta 20 características
    selector_f = SelectKBest(f_classif, k=k)
    selector_f.fit(X_no_corr, y)
    
    # Obtener puntuaciones
    f_scores = pd.DataFrame({
        'Feature': X_no_corr.columns,
        'F_Score': selector_f.scores_,
        'P_Value': selector_f.pvalues_
    })
    f_scores = f_scores.sort_values('F_Score', ascending=False)
    
    print("\nTop 10 características numéricas según ANOVA F-value:")
    print(f_scores.head(10))
    
    # Visualizar importancia de características según F-value
    plt.figure(figsize=(12, 8))
    sns.barplot(x='F_Score', y='Feature', data=f_scores.head(15))
    plt.title('Top 15 Características Numéricas según ANOVA F-value')
    plt.tight_layout()
    plt.show()
    
    # 3.4 Selección basada en Información Mutua para variables numéricas
    print("\n4. Selección basada en Información Mutua (solo variables numéricas):")
    
    # Aplicar SelectKBest con mutual_info_classif
    selector_mi = SelectKBest(mutual_info_classif, k=k)
    selector_mi.fit(X_no_corr, y)
    
    # Obtener puntuaciones
    mi_scores = pd.DataFrame({
        'Feature': X_no_corr.columns,
        'MI_Score': selector_mi.scores_
    })
    mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
    
    print("\nTop 10 características numéricas según Información Mutua:")
    print(mi_scores.head(10))
    
    # Visualizar importancia de características según Información Mutua
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_scores.head(15))
    plt.title('Top 15 Características Numéricas según Información Mutua')
    plt.tight_layout()
    plt.show()
    
    # 3.5 Selección basada en Random Forest (solo variables numéricas)
    print("\n5. Selección basada en Random Forest (solo variables numéricas):")
    
    # Entrenar un Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_no_corr, y)
    
    # Obtener importancia de características
    rf_importance = pd.DataFrame({
        'Feature': X_no_corr.columns,
        'Importance': rf.feature_importances_
    })
    rf_importance = rf_importance.sort_values('Importance', ascending=False)
    
    print("\nTop 10 características numéricas según Random Forest:")
    print(rf_importance.head(10))
    
    # Visualizar importancia de características según Random Forest
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_importance.head(15))
    plt.title('Top 15 Características Numéricas según Random Forest')
    plt.tight_layout()
    plt.show()
    
    # 3.6 Recursive Feature Elimination (RFE) para variables numéricas
    # print("\n6. Recursive Feature Elimination (RFE) (solo variables numéricas):")
    
    # # Aplicar RFE con Regresión Logística
    # rfe_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    # rfe = RFE(estimator=rfe_model, n_features_to_select=k, step=1)
    # rfe.fit(X_no_corr, y)
    
    # # Obtener ranking de características
    # rfe_ranking = pd.DataFrame({
    #     'Feature': X_no_corr.columns,
    #     'Ranking': rfe.ranking_
    # })
    # rfe_ranking = rfe_ranking.sort_values('Ranking')
    
    # print("\nTop 10 características numéricas según RFE:")
    # print(rfe_ranking.head(10))
    
    # 3.7 Análisis de características categóricas
    print("\n7. Análisis de características categóricas:")
    
    if len(categorical_columns) > 0:
        # Crear variables dummy para características categóricas
        X_cat = pd.get_dummies(X[categorical_columns], drop_first=True)
        print(f"Se generaron {X_cat.shape[1]} variables dummy a partir de {len(categorical_columns)} variables categóricas")
        
        # Evaluar importancia de variables dummy individuales
        if X_cat.shape[1] > 0:
            try:
                # Crear un modelo para evaluar la importancia
                cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
                cat_model.fit(X_cat, y)
                
                # Obtener importancia de cada variable dummy
                cat_importance = pd.DataFrame({
                    'Feature': X_cat.columns,
                    'Importance': cat_model.feature_importances_
                })
                cat_importance = cat_importance.sort_values('Importance', ascending=False)
                
                print("\nTop 10 variables dummy según Random Forest:")
                print(cat_importance.head(10))
                
                # Visualizar si hay suficientes características
                if len(cat_importance) >= 5:
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='Importance', y='Feature', data=cat_importance.head(10))
                    plt.title('Top 10 Variables Dummy según Random Forest')
                    plt.tight_layout()
                    plt.show()
                
                # Seleccionar las variables dummy más importantes
                # Determinar el umbral de importancia (p.ej., top 25% o un valor mínimo)
                importance_threshold = max(
                    cat_importance['Importance'].quantile(0.75),  # Top 25%
                    cat_importance['Importance'].mean()           # Media de importancia
                )
                
                # Seleccionar las variables dummy por encima del umbral
                selected_dummies = cat_importance[cat_importance['Importance'] > importance_threshold]['Feature'].tolist()
                
                print(f"\nSe seleccionaron {len(selected_dummies)} variables dummy con importancia > {importance_threshold:.6f}")
                
                # Mapear variables dummy a sus variables categóricas originales
                dummy_to_original = {}
                for dummy in selected_dummies:
                    # Extraer el nombre de la variable original (antes del primer '_')
                    for cat_col in categorical_columns:
                        if dummy.startswith(f"{cat_col}_"):
                            if cat_col not in dummy_to_original:
                                dummy_to_original[cat_col] = []
                            dummy_to_original[cat_col].append(dummy)
                
                # Seleccionar variables categóricas originales con dummies importantes
                selected_cat_features = list(dummy_to_original.keys())
                
                print(f"\nVariables categóricas con dummies importantes: {len(selected_cat_features)}")
                for cat_col, dummies in dummy_to_original.items():
                    print(f"  - {cat_col}: {len(dummies)} dummies importantes")
                
                # Almacenar la información para uso posterior
                top_cat_features = selected_cat_features
                top_dummy_features = selected_dummies
                
            except Exception as e:
                print(f"Error al evaluar características categóricas: {e}")
                top_cat_features = []
                top_dummy_features = []
        else:
            top_cat_features = []
            top_dummy_features = []
    else:
        print("No se encontraron características categóricas para analizar.")
        top_cat_features = []
        top_dummy_features = []
    
    # 3.8 Combinar resultados para identificar las características más importantes
    print("\n8. Características seleccionadas por múltiples métodos:")
    
    # Obtener top 15 características numéricas de cada método
    top_f = set(f_scores.head(15)['Feature'])
    top_mi = set(mi_scores.head(15)['Feature'])
    top_rf = set(rf_importance.head(15)['Feature'])
    # top_rfe = set(rfe_ranking.head(15)['Feature'])
    
    # Características numéricas que aparecen en al menos 2 métodos
    common_features = []
    for feature in X_no_corr.columns:
        count = sum([
            feature in top_f,
            feature in top_mi,
            feature in top_rf,
            # feature in top_rfe
        ])
        if count >= 2:  # Bajamos el umbral a 2 para ser más inclusivos
            common_features.append((feature, count))
    
    common_features = sorted(common_features, key=lambda x: x[1], reverse=True)
    
    print("\nCaracterísticas numéricas seleccionadas por múltiples métodos:")
    for feature, count in common_features:
        print(f"  - {feature}: seleccionada por {count} métodos")
    
    # Crear una lista final de características numéricas importantes
    selected_num_features = [feature for feature, _ in common_features]
    
    # Si hay menos de 10 características numéricas seleccionadas, agregar las más importantes según RF
    if len(selected_num_features) < 10:
        remaining = 10 - len(selected_num_features)
        for feature in rf_importance['Feature']:
            if feature not in selected_num_features:
                selected_num_features.append(feature)
                remaining -= 1
                if remaining == 0:
                    break
    
    # Definir el número máximo de características que queremos en total
    max_total_features = 50  # Ajusta este valor según tus necesidades
    
    # Calcular cuántas variables dummy podemos incluir
    max_dummy_features = max(10, max_total_features - len(selected_num_features))
    
    # Seleccionar las mejores variables dummy (si existen)
    selected_dummy_features = []
    if 'top_dummy_features' in locals() and len(top_dummy_features) > 0:
        # Limitar al número máximo establecido
        selected_dummy_features = top_dummy_features[:max_dummy_features]
    
    # Importante: NO incluimos las variables categóricas originales, solo sus dummies
    # Notar que top_cat_features solo se usa para referencia/información
    
    # Combinar características numéricas y variables dummy seleccionadas
    final_selected_features = {
        'numeric': selected_num_features,
        'dummy': selected_dummy_features
    }
    
    print(f"\nLista final de características seleccionadas:")
    print(f"  - {len(selected_num_features)} características numéricas")
    print(f"  - {len(selected_dummy_features)} variables dummy")
    print(f"  - Total: {len(selected_num_features) + len(selected_dummy_features)} características")
    
    print("\nCaracterísticas numéricas seleccionadas:")
    for feature in selected_num_features:
        print(f"  - {feature}")
    
    if selected_dummy_features:
        print("\nVariables dummy seleccionadas:")
        for feature in selected_dummy_features:
            print(f"  - {feature}")
    
    return final_selected_features

# 4. Preparación de datos para modelado
def prepare_data_for_modeling(df, selected_features=None):
    print("\n=== PREPARACIÓN DE DATOS PARA MODELADO ===")
    
    # Crear una copia para evitar modificar el original
    df_prep = df.copy()
    
    # 4.1 Manejo de valores faltantes
    print("\n1. Manejo de valores faltantes:")
    
    missing_values = df_prep.isnull().sum()
    missing_percentage = (missing_values / len(df_prep)) * 100
    
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    # Si hay valores faltantes, los rellenamos
    if columns_with_missing:
        print(f"Columnas con valores faltantes: {len(columns_with_missing)}")
        
        for col in columns_with_missing:
            missing_count = missing_values[col]
            missing_pct = missing_percentage[col]
            
            print(f"  - {col}: {missing_count} valores faltantes ({missing_pct:.2f}%)")
            
            # Verificar si la columna es numérica examinando sus valores no nulos
            non_null_values = df_prep[col].dropna()
            
            if len(non_null_values) > 0:
                # Intentar convertir a numérico para verificar el tipo
                try:
                    # Si es un string que contiene comas como separador decimal
                    if non_null_values.dtype == 'object':
                        sample_val = non_null_values.iloc[0]
                        if isinstance(sample_val, str) and ',' in sample_val:
                            # Convertir comas a puntos y luego a float
                            df_prep[col] = df_prep[col].str.replace(',', '.').astype(float)
                            is_numeric = True
                        else:
                            # Intentar conversión directa
                            pd.to_numeric(non_null_values, errors='raise')
                            is_numeric = True
                    else:
                        is_numeric = pd.api.types.is_numeric_dtype(non_null_values)
                except:
                    is_numeric = False
            else:
                is_numeric = False
            
            # Rellenar valores según el tipo de dato
            if is_numeric or df_prep[col].dtype in ['int64', 'float64']:
                # Para variables numéricas, usar mediana
                if is_numeric and df_prep[col].dtype == 'object':
                    # Convertir a numérico primero
                    df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
                
                median_value = df_prep[col].median()
                df_prep[col].fillna(median_value, inplace=True)
                print(f"    → Rellenado con la mediana: {median_value}")
            else:
                # Para variables categóricas, usar moda
                mode_value = df_prep[col].mode()[0]
                df_prep[col].fillna(mode_value, inplace=True)
                print(f"    → Rellenado con la moda: {mode_value}")
    else:
        print("No se detectaron valores faltantes.")
    
    # 4.2 Manejo de outliers
    print("\n2. Manejo de outliers:")
    print("Opciones para manejar outliers:")
    print("  a) Mantener los outliers: útil si representan casos reales y valiosos.")
    print("  b) Recortar (capping): limitar valores extremos a un umbral (por ejemplo, percentil 1-99).")
    print("  c) Eliminar filas con outliers: útil si son errores de datos.")
    print("  d) Transformar variables: usar transformaciones como logaritmo para reducir el impacto.")
    print("  e) Usar métodos robustos: algoritmos menos sensibles a outliers.")
    
    # Implementamos la opción de recorte (capping) como ejemplo
    print("\nAplicando recorte (capping) a variables numéricas con outliers:")
    
    # Seleccionamos solo columnas numéricas
    numeric_columns = df_prep.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'churn' in numeric_columns:
        numeric_columns.remove('churn')
    if 'Customer_ID' in numeric_columns:
        numeric_columns.remove('Customer_ID')
    
    # Seleccionamos columnas con más del 2% de outliers
    for col in numeric_columns:
        try:
            # Calcular percentiles para capping
            p01 = df_prep[col].quantile(0.01)
            p99 = df_prep[col].quantile(0.99)
            
            # Contar outliers antes del capping
            outliers_mask = (df_prep[col] < p01) | (df_prep[col] > p99)
            outliers_count = outliers_mask.sum()
            outliers_percentage = (outliers_count / len(df_prep)) * 100
            
            if outliers_percentage > 2:  # Si hay más del 2% de outliers
                print(f"  - {col}: {outliers_count} outliers ({outliers_percentage:.2f}%)")
                
                # Aplicar capping
                df_prep[col] = np.where(df_prep[col] < p01, p01, df_prep[col])
                df_prep[col] = np.where(df_prep[col] > p99, p99, df_prep[col])
                
                print(f"    → Valores limitados al rango [{p01:.2f}, {p99:.2f}]")
        except Exception as e:
            print(f"  - Error al procesar outliers en {col}: {e}")
    
    # 4.3 Codificación de variables categóricas
    print("\n3. Codificación de variables categóricas:")
    
    # Identificar columnas categóricas (excluyendo Customer_ID)
    categorical_columns = df_prep.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Customer_ID' in categorical_columns:
        categorical_columns.remove('Customer_ID')
    
    if categorical_columns:
        print(f"Variables categóricas identificadas: {len(categorical_columns)}")
        for col in categorical_columns:
            unique_values = df_prep[col].nunique()
            print(f"  - {col}: {unique_values} valores únicos")
        
        # Aplicar One-Hot Encoding
        print("\nAplicando One-Hot Encoding...")
        df_encoded = pd.get_dummies(df_prep, columns=categorical_columns, drop_first=True)
        
        print(f"Dimensiones antes de la codificación: {df_prep.shape}")
        print(f"Dimensiones después de la codificación: {df_encoded.shape}")
    else:
        print("No se detectaron variables categóricas.")
        df_encoded = df_prep.copy()
    
    # 4.4 Escalado de características
    print("\n4. Escalado de características:")
    
    # Separar características y variable objetivo
    X = df_encoded.drop(['churn', 'Customer_ID'], axis=1, errors='ignore')
    y = df_encoded['churn']
    
    # Si se proporcionaron características seleccionadas, usarlas
    if selected_features:
        # Extraer las listas de características seleccionadas
        selected_numeric = selected_features.get('numeric', [])
        selected_dummy = selected_features.get('dummy', [])
        
        # Verificar que las características existen en el dataframe
        valid_numeric = [f for f in selected_numeric if f in X.columns]
        valid_dummy = [f for f in selected_dummy if f in X.columns]
        
        # Combinar todas las características válidas
        final_features = valid_numeric + valid_dummy
        
        print(f"Utilizando {len(final_features)} características seleccionadas:")
        print(f"  - {len(valid_numeric)} características numéricas")
        print(f"  - {len(valid_dummy)} variables dummy")
        
        # Filtrar el dataframe para incluir solo las características seleccionadas
        X = X[final_features]
    
    print(f"Dimensiones finales del conjunto de características: {X.shape}")
    
    # Aplicar escalado robusto (menos sensible a outliers) solo a variables numéricas
    print("\nAplicando escalado robusto a variables numéricas...")
    
    # Identificar columnas numéricas en X
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Crear un scaler y aplicarlo solo a columnas numéricas
        scaler = RobustScaler()
        X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
        
        # Crear un nuevo DataFrame con las columnas numéricas escaladas
        X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)
        
        # Reemplazar las columnas numéricas originales con las escaladas
        X_scaled_df = X.copy()
        X_scaled_df[numeric_cols] = X_numeric_scaled_df
    else:
        # Si no hay columnas numéricas, mantener X como está
        X_scaled_df = X.copy()
        print("No se encontraron columnas numéricas para escalar.")
    
    print("\nPrimeras filas de los datos escalados:")
    print(X_scaled_df.head())
    
    # 4.5 División en conjuntos de entrenamiento y prueba
    print("\n5. División en conjuntos de entrenamiento y prueba:")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    print(f"Distribución de clases en entrenamiento: {y_train.value_counts(normalize=True)}")
    print(f"Distribución de clases en prueba: {y_test.value_counts(normalize=True)}")
    
    # 4.6 Verificación rápida con un modelo simple
    print("\n6. Verificación rápida con un modelo simple:")
    
    # Entrenar un modelo de Regresión Logística
    try:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"AUC-ROC con Regresión Logística: {auc_score:.4f}")
        
        # Coeficientes del modelo
        coefficients = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': model.coef_[0]
        })
        
        coefficients = coefficients.sort_values('Coefficient', ascending=False)
        
        print("\nCoeficientes del modelo de Regresión Logística:")
        print(coefficients.head(10))  # Top 10 características positivas
        print("\n...")
        print(coefficients.tail(10))  # Top 10 características negativas
        
        # Visualizar coeficientes
        plt.figure(figsize=(12, 8))
        top_features = pd.concat([coefficients.head(10), coefficients.tail(10)])
        sns.barplot(x='Coefficient', y='Feature', data=top_features)
        plt.title('Principales Coeficientes de Regresión Logística')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al entrenar el modelo de verificación: {e}")
        model = None
    
    return X_train, X_test, y_train, y_test, model

# Función principal para ejecutar todo el análisis
def main(file_path):
    # 1. Cargar datos
    df = load_data(file_path)
    
    # 2. Analizar outliers
    outliers_summary, outlier_counts = analyze_outliers(df)
    
    # 3. Seleccionar características
    selected_features = feature_selection(df)
    
    # 4. Preparar datos para modelado
    X_train, X_test, y_train, y_test, initial_model = prepare_data_for_modeling(df, selected_features)
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("\nLos datos están listos para el modelado predictivo.")
    print("Próximos pasos recomendados:")
    print("1. Ajustar hiperparámetros de los modelos")
    print("2. Probar diferentes algoritmos (Random Forest, XGBoost, etc.)")
    print("3. Evaluar modelos con métricas completas")
    print("4. Interpretar resultados para el negocio")
    
    return df, selected_features, X_train, X_test, y_train, y_test, initial_model

# Ejecutar el análisis completo
if __name__ == "__main__":
    file_path = '../SDG-PruebaTecnica/dataset.csv'
    df, selected_features, X_train, X_test, y_train, y_test, initial_model = main(file_path)
