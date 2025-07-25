# MLOps--MLflow--Cloud_Machine_Learning_en_AWS_usando_MLflow

Este script realiza clasificación binaria utilizando XGBoost en Spark con seguimiento de MLflow.
Lee un archivo Parquet, entrena un modelo, lo evalúa y registra resultados en MLflow.
El script está listo para ser ejecutado en AWS EMR con `spark-submit` y manejará grandes volúmenes de datos gracias a la implementación distribuida de XGBoost para Spark.

```
## Importación de bibliotecas
Importación de módulos esenciales para procesamiento distribuido, machine learning, 
seguimiento de experimentos y visualización de resultados
```
import pyspark                                                        # Biblioteca principal para computación distribuida
from pyspark.sql import SparkSession                                  # Punto de entrada para DataFrames y SQL
from pyspark.ml.feature import VectorAssembler                        # Para combinar características en un vector
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator  # Evaluadores de modelos
from xgboost.spark import XGBoostClassifier                           # Clasificador XGBoost distribuido para Spark
import mlflow                                                         # Plataforma para seguimiento de experimentos
import mlflow.pyspark.ml                                              # Integración de MLflow con PySpark
import matplotlib.pyplot as plt                                       # Visualización de gráficos
import seaborn as sns                                                 # Visualización estadística mejorada
from sklearn.metrics import confusion_matrix, roc_curve, auc          # Métricas de evaluación
import numpy as np                                                    # Cómputo numérico eficiente
```
## Configuración del entorno Spark
Creación de una sesión Spark para procesamiento distribuido. 
'appName' define el nombre visible en la interfaz de Spark UI
```
spark = SparkSession.builder.appName('XGBoost Classification').getOrCreate()  # Inicializa el entorno Spark
```
## Carga y preparación de datos
Lectura del dataset en formato Parquet (optimizado para big data) y transformación a formato adecuado para algoritmos ML
```
df = spark.read.parquet('data.parquet')                               # Carga datos desde archivo Parquet a DataFrame
```
## Identificación automática de columnas de características
```
features = [col for col in df.columns if col != 'target']             # Lista todas las columnas excepto el objetivo
```
## Conversión de características a formato vectorial requerido por XGBoost
```
assembler = VectorAssembler(inputCols=features, outputCol='features') # Configura ensamblador de vectores
df = assembler.transform(df)                                          # Transforma DataFrame añadiendo columna 'features'
```
# División de datos para entrenamiento y validación
Creación de conjuntos de entrenamiento y validación mediante muestreo estratificado
Semilla fija para reproducibilidad de resultados
```
train_df, val_df = df.randomSplit([0.8, 0.2], seed=123)               # Divide datos: 80% entrenamiento, 20% validación
```
## Configuración de experimento MLflow
MLflow permite rastrear parámetros, métricas y artefactos del modelo
para facilitar reproducibilidad y comparación de experimentos
```
mlflow.set_experiment('New Experiment')                               # Crea/reutiliza experimento 'New Experiment'

mlflow.pyspark.ml.autolog()                                           # Autoregistro de parámetros y métricas del modelo
```
## Entrenamiento y evaluación del modelo

Bloque principal donde se define, entrena y evalúa el modelo XGBoost con registro completo en MLflow
```
with mlflow.start_run(run_name='xgboost_model_classification') as run:
    # Configuración del clasificador XGBoost para problema binario
    xgb_clf = XGBoostClassifier(
        featuresCol='features',          # Columna vectorial de características
        labelCol='target',               # Variable objetivo
        objective='binary:logistic',     # Objetivo de clasificación binaria
        eval_metric='logloss',           # Métrica de evaluación durante entrenamiento
        num_round=100,                   # Número de iteraciones de boosting
        # Parámetros adicionales (ej. max_depth, eta) pueden añadirse aquí
    )
    
    # Entrenamiento del modelo
    model = xgb_clf.fit(train_df)        # Ajusta modelo a datos de entrenamiento
    
    # Generación de predicciones
    preds = model.transform(val_df)      # Predice sobre conjunto de validación
    
    # ----------------------------------------------------------------------------------------
    # Cálculo y registro de métricas de evaluación
    # ----------------------------------------------------------------------------------------
    # Precisión (Accuracy)
    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(preds)             # Calcula precisión
    mlflow.log_metric('accuracy', accuracy)          # Registra en MLflow
    
    # Área bajo curva ROC (AUC)
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol='target',
        rawPredictionCol='probability',              # Usa columna de probabilidades
        metricName='areaUnderROC')
    auc_score = evaluator_auc.evaluate(preds)        # Calcula AUC
    mlflow.log_metric('auc', auc_score)              # Registra en MLflow
    
    # Precisión ponderada (Weighted Precision)
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol='target',
        predictionCol='prediction',
        metricName='weightedPrecision')
    precision = evaluator_precision.evaluate(preds)  # Calcula precisión
    mlflow.log_metric('precision', precision)        # Registra en MLflow
    
    # Recall ponderado (Weighted Recall)
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol='target',
        predictionCol='prediction',
        metricName='weightedRecall')
    recall = evaluator_recall.evaluate(preds)        # Calcula recall
    mlflow.log_metric('recall', recall)              # Registra en MLflow
    
    # Puntuación F1
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol='target',
        predictionCol='prediction',
        metricName='f1')
    f1 = evaluator_f1.evaluate(preds)                # Calcula F1
    mlflow.log_metric('f1', f1)                      # Registra en MLflow
    
    # ----------------------------------------------------------------------------------------
    # Generación de visualizaciones
    # ----------------------------------------------------------------------------------------
    # Conversión de resultados a arrays de NumPy para visualización
    # (Asume que el conjunto de validación es manejable en tamaño)
    labels = np.array([row['target'] for row in preds.select('target').collect()])                        # Obtiene etiquetas reales
    predictions = np.array([row['prediction'] for row in preds.select('prediction').collect()])           # Obtiene predicciones
    probabilities = np.array([row['probability'][1] for row in preds.select('probability').collect()])    # Probabilidades clase positiva
    
    # Matriz de confusión
    cm = confusion_matrix(labels, predictions)             # Calcula matriz
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')     # Visualiza con anotaciones
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')                    # Guarda imagen
    mlflow.log_artifact('confusion_matrix.png')            # Registra artefacto en MLflow
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(labels, probabilities)         # Calcula puntos de la curva
    roc_auc = auc(fpr, tpr)                                # Calcula área bajo curva
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k—')                         # Línea de referencia
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png')                           # Guarda imagen
    mlflow.log_artifact('roc_curve.png')                   # Registra artefacto en MLflow
```


