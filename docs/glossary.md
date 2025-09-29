## Glosario técnico

### GAN (Generative Adversarial Network)

Arquitectura compuesta por dos redes neuronales que se entrenan de forma competitiva:

- Generador: produce muestras sintéticas a partir de ruido aleatorio (vector latente).
- Discriminador: intenta distinguir entre muestras reales y sintéticas.

Durante el entrenamiento, el generador mejora para engañar al discriminador, mientras el discriminador mejora para no ser engañado. Este proceso adversario (competitivo) guía al generador hacia muestras cada vez más realistas. En implementaciones tabulares, las salidas del generador suelen normalizarse a un rango fijo (por ejemplo, [-1, 1] con `Tanh`) y luego desnormalizarse al dominio original.

### Entrenamiento estratificado (stratified training)

Estrategia en la que el modelo se entrena por subconjuntos según la clase del objetivo. En tabulares, se entrena un generador por cada clase, lo que favorece que el modelo aprenda las distribuciones condicionadas (condicionales) específicas de cada clase.

### Submuestreo (undersampling)

Técnica de balanceo que reduce el número de registros de la clase mayoritaria hasta igualarlo (o aproximarlo) al de la clase minoritaria. Ayuda a mitigar desbalances que pueden sesgar el entrenamiento.

### One‑Hot Encoding

Codificación de variables categóricas en columnas binarias (0/1), una por categoría. Permite usar modelos que esperan entradas numéricas.

### Estandarización (StandardScaler)

Transformación que lleva una variable a media 0 y desviación estándar 1. Suele mejorar la estabilidad numérica en entrenamiento.

### Normalización a [-1, 1]

Escalado lineal que mapea valores al intervalo [-1, 1]. Es habitual cuando el generador usa `Tanh`, cuya salida está precisamente en ese rango.

### Vector latente (latent vector)

Entrada aleatoria del generador que codifica factores abstractos de variación. Al muestrearlo repetidamente se obtienen diferentes muestras sintéticas.

### Función de pérdida BCELoss (Binary Cross‑Entropy)

Medida de discrepancia entre predicciones probabilísticas y etiquetas binarias. En GANs básicos, el discriminador y el generador se optimizan con variantes de esta pérdida.

### Optimizador Adam

Algoritmo de descenso de gradiente estocástico con momentos adaptativos. Suele converger más rápido y de forma estable que SGD clásico en redes profundas.

### Bins (histograma)

Particiones del rango de una variable continua para construir histogramas. Comparar distribuciones con los mismos bins permite una comparación justa.

### Diferencia relativa de la media (rel_mean_diff)

Definición: \( |\mu*{syn} - \mu*{real}| / (|\mu\_{real}| + 1e{-8}) \).
Interpretación: valores bajos indican medias similares entre datos reales y sintéticos.

### Diferencia relativa de la desviación estándar (std_rel_diff)

Definición: \( |\sigma*{syn} - \sigma*{real}| / \sigma\_{real} \).
Interpretación: valores bajos indican dispersión (variabilidad) similar.

### Fidelidad (en datos sintéticos)

Grado de similitud estadística entre los datos sintéticos y los reales, medida con métricas como `rel_mean_diff`, `std_rel_diff` y comparativas visuales.

### Drift (deriva)

Cambio sistemático entre distribuciones. En este contexto, se detecta cuando la media o la desviación estándar del sintético se alejan de las del real más allá de un umbral predefinido.
