# Glosario técnico

## Términos fundamentales del proyecto

### GAN (Generative Adversarial Network)

Arquitectura compuesta por dos redes neuronales que se entrenan de forma competitiva:

- **Generador**: produce muestras sintéticas a partir de ruido aleatorio (vector latente)
- **Discriminador**: intenta distinguir entre muestras reales y sintéticas

Durante el entrenamiento, el generador mejora para engañar al discriminador, mientras el discriminador mejora para no ser engañado. En este proyecto se usa para generar datos sintéticos del dataset Titanic.

### Epoch (Época)

Una época es una pasada completa a través de todo el dataset de entrenamiento. En este proyecto se usan 1400 épocas para entrenar el GAN.

### Batch Size (Tamaño de Lote)

Número de muestras que se procesan en una sola iteración del entrenamiento. En este proyecto se usa batch_size=64.

### Latent Dimension (Dimensión Latente)

Tamaño del vector de ruido aleatorio que se usa como entrada para el generador. En este proyecto se usa latent_dim=100.

### Learning Rate (Tasa de Aprendizaje)

Parámetro que controla qué tan grandes son los pasos que da el optimizador durante el entrenamiento. En este proyecto se usa lr=0.0002.

## Técnicas de preprocesamiento utilizadas

### One-Hot Encoding

Técnica de codificación que convierte variables categóricas en vectores binarios. Cada categoría se representa como un vector con un 1 en la posición correspondiente y 0s en el resto. Ejemplo: "male" → [0,1], "female" → [1,0].

### StandardScaler

Transformación que normaliza variables numéricas restando la media y dividiendo por la desviación estándar. Esto resulta en variables con media=0 y desviación estándar=1.

### Normalización a [-1, 1]

Escalado lineal que mapea valores al intervalo [-1, 1]. Se usa porque el generador utiliza la función Tanh, cuya salida está en este rango.

### RandomUnderSampler

Técnica de balanceo de clases que reduce el número de muestras de la clase mayoritaria hasta igualar la clase minoritaria. Se usa para evitar que el modelo se sesgue hacia la clase más frecuente.

### Entrenamiento estratificado (stratified training)

Estrategia donde se entrena un modelo separado para cada clase del objetivo. En este proyecto se entrena un GAN por cada clase de supervivencia (sobrevivió/no sobrevivió).

**Configuración actual**:

- **Activado**: `STRATIFIED_TRAINING: True`
- **Épocas por clase**: 1400 épocas
- **Total de épocas**: 2800 épocas (1400 × 2 clases)
- **Muestras por clase**: 500 muestras sintéticas por clase
- **Balanceo perfecto**: 50%-50% en datos sintéticos

**Ventajas**:

- Preservación de dependencias específicas por clase
- Balanceo automático perfecto
- Mejor calidad en la generación de patrones por clase
- Evita que una clase domine el entrenamiento

### División de datos (Train/Test Split)

En este proyecto **se realiza división estándar 80-20**:

- **Datos de entrenamiento**: 80% (712 muestras)
- **Datos de prueba**: 20% (179 muestras)
- **Estrategia**: Entrenamiento solo con datos de entrenamiento balanceados (546 muestras)
- **Evaluación**: Comparación entre datos sintéticos generados (1,000) y datos reales de entrenamiento

## Métricas de evaluación implementadas

### Diferencia relativa de la media (rel_mean_diff)

Fórmula: |media_sintética - media_real| / (|media_real| + 1e-8)
Interpretación: valores bajos indican medias similares entre datos reales y sintéticos.

### Diferencia relativa de la desviación estándar (std_rel_diff)

Fórmula: |std_sintética - std_real| / std_real
Interpretación: valores bajos indican dispersión similar.

### Drift (deriva)

Cambio sistemático entre distribuciones. Se detecta cuando la media o la desviación estándar del sintético se alejan de las del real más allá de un umbral predefinido (30% para numéricas, 20% para categóricas).

### Porcentaje discriminador general

Métrica que indica qué tan bien el generador reproduce las distribuciones originales. Se calcula como: (variables_bien_generadas / total_variables) \* 100

**Resultado actual**: 77.78%

- **Total de variables**: 9
- **Variables con drift alto**: 2 (sibsp, parch)
- **Variables bien generadas**: 7

**Interpretación**:

- **> 80%**: Excelente capacidad discriminativa
- **60-80%**: Buena capacidad discriminativa (caso actual)
- **< 60%**: Capacidad discriminativa limitada

### Colapso de categorías

Fenómeno donde el generador produce casi exclusivamente una categoría dominante, ignorando las categorías minoritarias.

**Ejemplo anterior**: generar 99.7% de pasajeros del puerto S y solo 0.18% del puerto C.

**Mejora con entrenamiento estratificado**: El colapso se ha reducido significativamente:

- Puerto S: 85.1% (vs. 99.7% anterior)
- Puerto C: 14.9% (vs. 0.18% anterior)
- Puerto Q: 1.2% (vs. 0.08% anterior)

### Variables de conteo discreto

Variables que representan cantidades enteras (0, 1, 2, 3...), como número de hermanos (sibsp) o padres/hijos (parch). Son particularmente desafiantes para GANs simples debido a su naturaleza discreta.

**Problemas identificados en el proyecto**:

- **sibsp**: rel_mean_diff = 0.753 (75.3% de diferencia)
- **parch**: rel_mean_diff = 0.318 (31.8% de diferencia)

**Causas del problema**:

- Las GANs generan valores continuos que se redondean a enteros
- Las distribuciones de conteo son difíciles de aprender para redes simples
- El entrenamiento estratificado ayuda pero no resuelve completamente el problema

## Funciones de activación utilizadas

### Tanh Activation (Tangente Hiperbólica)

Función de activación no lineal que produce salidas en el rango [-1, 1].

**Fórmula**: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Características**:

- Rango de salida: [-1, 1]
- Es simétrica alrededor del origen (0, 0)
- Tiene gradientes suaves, evitando el problema de gradientes que se desvanecen
- Se usa en la capa de salida del generador porque los datos se normalizan a [-1, 1]

**En el proyecto**: Se usa en la última capa del generador para producir valores sintéticos en el rango correcto.

### ReLU Activation (Rectified Linear Unit)

Función de activación que devuelve 0 para valores negativos y el valor original para valores positivos.

**Fórmula**: ReLU(x) = max(0, x)

**Características**:

- Rango de salida: [0, +∞)
- Es computacionalmente eficiente (solo requiere comparación)
- Evita el problema de gradientes que se desvanecen para valores positivos
- Puede causar "neuronas muertas" (gradiente 0 para valores negativos)
- Se usa en las capas ocultas del generador y discriminador

**En el proyecto**: Se usa en todas las capas ocultas para introducir no-linealidad y permitir que las redes aprendan patrones complejos.

## Optimización

### BCELoss (Binary Cross-Entropy Loss)

Función de pérdida que mide la diferencia entre predicciones probabilísticas y etiquetas binarias. Se usa para entrenar tanto el discriminador como el generador.

### Adam Optimizer

Algoritmo de optimización que adapta la tasa de aprendizaje para cada parámetro individualmente. Se usa para optimizar tanto el generador como el discriminador.

## Arquitectura modular y orquestadores

### Orquestador (Orchestrator)

Patrón de diseño donde un módulo principal coordina funciones especializadas de otros módulos. En este proyecto:

- **`analyzer.py`**: Orquestador de análisis de datos
- **`gan.py`**: Orquestador de generación GAN

### Módulos especializados

Archivos que contienen funciones específicas para una responsabilidad particular:

- **`plotting.py`**: Funciones de visualización
- **`balancing.py`**: Lógica de balanceo de clases
- **`comparison.py`**: Comparación real vs sintético
- **`metrics.py`**: Cálculo de métricas de evaluación
- **`gan_models.py`**: Arquitecturas de redes neuronales
- **`training.py`**: Lógica de entrenamiento
- **`data_processing.py`**: Preprocesamiento específico para GAN
- **`generation.py`**: Estrategias de generación

### Configuración centralizada

Sistema donde todos los parámetros del proyecto se definen en un solo lugar (`src/config.py`) para facilitar el mantenimiento y la consistencia.

### Validaciones robustas

Sistema de validación implementado en `src/validation.py` que asegura:

- Calidad de datos de entrada
- Parámetros de configuración válidos
- Tipos de datos correctos
- Manejo seguro de errores
