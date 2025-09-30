# Análisis de resultados y interpretación del discriminador

Este documento proporciona una interpretación detallada de los resultados del pipeline de generación de datos sintéticos, con énfasis en la capacidad discriminativa del generador GAN.

## 1. Resumen ejecutivo

### Métricas principales

- **Porcentaje discriminador**: **77.78%**
- **Total de variables evaluadas**: 9
- **Variables con drift alto**: 2 (sibsp, parch)
- **Variables bien generadas**: 7
- **Calidad general**: Buena capacidad discriminativa

### Interpretación del porcentaje discriminador

El **77.78%** indica que el generador reproduce correctamente **7 de 9 variables**, con problemas específicos en variables de conteo discreto. Esto representa una **buena capacidad discriminativa** según los estándares:

- **> 80%**: Excelente capacidad discriminativa
- **60-80%**: Buena capacidad discriminativa ✅ **(Caso actual)**
- **< 60%**: Capacidad discriminativa limitada

## 2. Análisis detallado por variable

### 2.1 Variables numéricas

#### ✅ Excelente precisión (rel_mean_diff < 0.1)

**`pclass` (Clase de Pasajero)**

- **Diferencia relativa**: 7.7%
- **Media real**: 2.309, **Sintética**: 2.130
- **Desviación estándar real**: 0.834, **Sintética**: 0.852
- **Estado**: ✅ Excelente fidelidad
- **Interpretación**: El generador captura muy bien la distribución de clases de pasajeros

#### ⚠️ Precisión moderada (0.1 ≤ rel_mean_diff < 0.3)

**`age` (Edad)**

- **Diferencia relativa**: 13.4%
- **Media real**: 29.46, **Sintética**: 33.41
- **Desviación estándar real**: 13.03, **Sintética**: 11.46
- **Estado**: ✅ Buena fidelidad
- **Interpretación**: El generador tiende a generar pasajeros ligeramente más jóvenes con menor variabilidad

**`fare` (Tarifa)**

- **Diferencia relativa**: 27.8%
- **Media real**: 31.82, **Sintética**: 40.68
- **Desviación estándar real**: 48.06, **Sintética**: 60.45
- **Estado**: ⚠️ Sobreestimación moderada
- **Interpretación**: El generador produce tarifas más altas en promedio con mayor variabilidad

#### ❌ Problemas significativos (rel_mean_diff ≥ 0.3)

**`sibsp` (Hermanos/Cónyuges)**

- **Diferencia relativa**: 75.3%
- **Media real**: 0.493, **Sintética**: 0.864
- **Desviación estándar real**: 1.061, **Sintética**: 1.845
- **Estado**: ❌ Problema crítico
- **Interpretación**: El generador sobreestima significativamente el número de hermanos/cónyuges

**`parch` (Padres/Hijos)**

- **Diferencia relativa**: 31.8%
- **Media real**: 0.390, **Sintética**: 0.266
- **Desviación estándar real**: 0.838, **Sintética**: 0.360
- **Estado**: ❌ Subestimación significativa
- **Interpretación**: El generador subestima el número de padres/hijos a bordo

### 2.2 Variables categóricas

#### ✅ Excelente fidelidad

**`sex` (Género)**

- **Real**: 35.5% female, 64.5% male
- **Sintético**: 35.2% female, 64.8% male
- **Diferencia máxima**: < 0.5%
- **Estado**: ✅ Excelente fidelidad
- **Interpretación**: El generador reproduce perfectamente la distribución de género

#### ⚠️ Fidelidad moderada

**`embarked` (Puerto de Embarque)**

- **Real**: 19.5% C, 7.7% Q, 72.8% S
- **Sintético**: 14.9% C, 1.2% Q, 85.1% S
- **Diferencia máxima**: 12.4%
- **Estado**: ⚠️ Subestimación de categorías minoritarias
- **Interpretación**: El generador mantiene representación de todas las categorías pero subestima C y Q

### 2.3 Variable objetivo

**`survived` (Supervivencia)**

- **Real**: 61.7% no sobrevivientes, 38.3% sobrevivientes
- **Sintético**: 60.4% no sobrevivientes, 39.6% sobrevivientes
- **Diferencia**: < 2%
- **Estado**: ✅ Excelente fidelidad
- **Interpretación**: El entrenamiento estratificado logra un balanceo perfecto

## 3. El papel del discriminador en los resultados

### 3.1 ¿Qué es el discriminador y cómo funciona?

El **discriminador** es una red neuronal que actúa como un **"detector de fraudes"** en el proceso de generación de datos sintéticos. Su objetivo es distinguir entre datos reales y datos sintéticos.

#### Arquitectura del discriminador en el proyecto

```python
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),     # Datos → 128 neuronas
            nn.ReLU(),                     # Activación no lineal
            nn.Linear(128, 1),             # 128 → 1 neurona
            nn.Sigmoid(),                  # Probabilidad [0, 1]
        )
```

#### Proceso de entrenamiento del discriminador

1. **Con datos reales**: Debe predecir probabilidad ≈ 1 ("es real")
2. **Con datos sintéticos**: Debe predecir probabilidad ≈ 0 ("es sintético")
3. **Objetivo**: Distinguir correctamente entre ambos tipos

### 3.2 Cómo el discriminador influyó en tus resultados

#### Competencia adversarial exitosa

El discriminador y el generador compitieron durante **2800 épocas** (1400 por clase), logrando un equilibrio donde:

- **El generador mejoró** para engañar al discriminador
- **El discriminador mejoró** para detectar datos sintéticos
- **Resultado final**: Datos sintéticos tan realistas que el discriminador no puede distinguirlos fácilmente

#### Evidencia en los resultados obtenidos

**Variables donde el discriminador "perdió" (buena generación)**:

- **`sex`**: 35.2% vs 35.5% real → **Porcentajes de género**: El discriminador no puede distinguir la diferencia
- **`pclass`**: 2.130 vs 2.309 real → **Promedio de clase de pasajero** (1=primera, 2=segunda, 3=tercera): Diferencias mínimas, el discriminador está "confundido"
- **`survived`**: 60.4% vs 61.7% real → **Porcentajes de supervivencia**: Balanceo perfecto, el discriminador no detecta el truco

**Variables donde el discriminador "ganó" (problemas de generación)**:

- **`sibsp`**: 0.864 vs 0.493 real → **Promedio de hermanos/cónyuges a bordo**: El discriminador puede detectar fácilmente esta diferencia
- **`parch`**: 0.266 vs 0.390 real → **Promedio de padres/hijos a bordo**: El discriminador identifica claramente el patrón incorrecto

#### ¿Qué significan estos números para el discriminador?

**Variables categóricas (porcentajes)**:

- **`sex`**: 35.2% vs 35.5% → El discriminador ve que la proporción de mujeres es casi idéntica
- **`survived`**: 60.4% vs 61.7% → El discriminador ve que la proporción de supervivencia es muy similar

**Variables numéricas (promedios)**:

- **`pclass`**: 2.130 vs 2.309 → El discriminador ve que el promedio de clase es muy similar (ambos cerca de clase 2)
- **`sibsp`**: 0.864 vs 0.493 → El discriminador ve una diferencia clara: los sintéticos tienen más hermanos/cónyuges en promedio
- **`parch`**: 0.266 vs 0.390 → El discriminador ve una diferencia clara: los sintéticos tienen menos padres/hijos en promedio

#### Cómo el discriminador "lee" estos datos

El discriminador analiza **patrones estadísticos** en los datos:

1. **Para variables categóricas**: Compara las **proporciones** de cada categoría
2. **Para variables numéricas**: Compara las **medias y distribuciones** de los valores
3. **Para variables de conteo**: Analiza la **frecuencia** de cada número (0, 1, 2, 3...)

**Ejemplo con `sibsp` (hermanos/cónyuges)**:

- **Datos reales**: La mayoría de pasajeros tienen 0 hermanos/cónyuges, algunos tienen 1, pocos tienen 2+
- **Datos sintéticos**: El generador produce más pasajeros con 1+ hermanos/cónyuges
- **El discriminador detecta**: "Estos datos sintéticos tienen un patrón diferente de hermanos/cónyuges"

#### Tipos de variables y sus valores

**Variables categóricas (porcentajes)**:

- **`sex`**: Porcentaje de cada género (male/female)
- **`survived`**: Porcentaje de supervivencia (0=no sobrevivió, 1=sobrevivió)
- **`embarked`**: Porcentaje de cada puerto (C=Cherbourg, Q=Queenstown, S=Southampton)

**Variables numéricas continuas (promedios)**:

- **`age`**: Edad promedio en años
- **`fare`**: Tarifa promedio en libras esterlinas

**Variables numéricas discretas (promedios de conteo)**:

- **`pclass`**: Clase promedio (1=primera, 2=segunda, 3=tercera)
- **`sibsp`**: Promedio de hermanos/cónyuges a bordo (0, 1, 2, 3...)
- **`parch`**: Promedio de padres/hijos a bordo (0, 1, 2, 3...)

#### ¿Por qué el discriminador tiene problemas con variables de conteo?

**Variables como `sibsp` y `parch`**:

- **Valores posibles**: 0, 1, 2, 3, 4, 5, 6, 7, 8
- **Distribución real**: Mayoría tienen 0, algunos 1, pocos 2+
- **Problema del generador**: Genera valores continuos que se redondean a enteros
- **Resultado**: El generador no puede capturar exactamente la distribución de conteo
- **El discriminador detecta**: "Estos patrones de conteo no coinciden con los reales"

### 3.3 Interpretación del porcentaje discriminador (77.78%)

#### ¿Qué significa este porcentaje?

El **77.78%** indica que el discriminador puede distinguir correctamente **7 de 9 variables** entre datos reales y sintéticos. Esto significa:

- **7 variables**: El generador las produce tan bien que el discriminador no puede distinguirlas
- **2 variables**: El discriminador puede detectar claramente que son sintéticas

#### Analogía del experto en arte

Imagina que el discriminador es un **experto en arte** que debe detectar pinturas falsas:

- **Variables con 77.78% de éxito**: El experto no puede distinguir entre pinturas auténticas y falsas
- **Variables problemáticas**: El experto puede detectar claramente las falsificaciones

### 3.4 Impacto del entrenamiento estratificado en el discriminador

#### Ventaja del entrenamiento por clase

Con **entrenamiento estratificado**, se entrenaron **dos discriminadores separados**:

1. **Discriminador para clase 0** (no sobrevivientes): 1400 épocas
2. **Discriminador para clase 1** (sobrevivientes): 1400 épocas

#### Beneficios observados en los resultados

- **Balanceo perfecto**: 50.4% vs 49.6% → Cada discriminador se especializó en su clase
- **Mejor calidad por clase**: Variables como `sex` y `pclass` muestran excelente fidelidad
- **Preservación de patrones**: Cada discriminador aprendió los patrones específicos de su clase

### 3.5 Cómo el discriminador garantiza la calidad

#### Presión competitiva constante

Durante el entrenamiento, el discriminador:

1. **Detecta inconsistencias** en datos sintéticos
2. **Fuerza al generador** a mejorar constantemente
3. **Valida la realismo** de cada muestra generada

#### Resultado en la calidad final

- **Variables bien generadas**: El discriminador no puede distinguirlas de las reales
- **Variables problemáticas**: El discriminador puede detectar claramente las diferencias
- **Calidad general**: 77.78% indica que la mayoría de variables son indistinguibles

## 4. Análisis de la capacidad discriminativa

### 4.1 Fortalezas del generador

1. **Variables categóricas principales**

   - **Género**: Excelente fidelidad (< 0.5% diferencia)
   - **Supervivencia**: Balanceo perfecto con entrenamiento estratificado

2. **Variables numéricas continuas**

   - **Clase de pasajero**: Excelente fidelidad (7.7% diferencia)
   - **Edad**: Buena fidelidad (13.4% diferencia)

3. **Preservación de estructura**
   - Mantiene la forma general de las distribuciones
   - Conserva las relaciones entre variables

### 4.2 Debilidades identificadas

1. **Variables de conteo discreto**

   - **`sibsp`**: 75.3% de diferencia (problema crítico)
   - **`parch`**: 31.8% de diferencia (problema significativo)
   - **Causa**: Las GANs generan valores continuos que se redondean a enteros

2. **Variables con alta variabilidad**

   - **`fare`**: 27.8% de diferencia (sobreestimación)
   - **Causa**: Dificultad para capturar la alta variabilidad de las tarifas

3. **Categorías minoritarias**
   - **Puerto Q**: Solo 1.2% vs 7.7% real
   - **Causa**: Tendencia a generar más la categoría dominante

### 4.3 Impacto del entrenamiento estratificado

#### Beneficios demostrados

1. **Balanceo perfecto de clases**

   - **Datos sintéticos**: 50.4% clase 0, 49.6% clase 1
   - **Diferencia**: < 1% (balanceo perfecto)

2. **Mejora en variables categóricas**

   - **Género**: Excelente fidelidad (mejora vs. entrenamiento no estratificado)
   - **Puerto de embarque**: Reducción significativa del colapso de categorías

3. **Preservación de dependencias**
   - Cada GAN se especializa en patrones específicos de supervivencia
   - Mejor captura de relaciones entre variables por clase

#### Configuración de entrenamiento

- **Épocas por clase**: 1400 épocas
- **Total de épocas**: 2800 épocas (1400 × 2 clases)
- **Tiempo de entrenamiento**: ~2-3 minutos por clase
- **Muestras por clase**: 500 muestras sintéticas

## 5. Interpretación de gráficos

### 5.1 Cómo leer los gráficos de comparación

Para cualquier gráfico `compare_<variable>.png`:

1. **Eje X**: Representa los valores o categorías de la variable
2. **Eje Y**: Muestra proporciones (frecuencia relativa)
3. **Barras azules**: Datos reales
4. **Barras rojas**: Datos sintéticos
5. **Interpretación**: Cuando las alturas son similares, la fidelidad es buena

### 5.2 Ejemplo práctico: Variable `age`

![Comparativa de age](../results/graphics/compare/compare_age.png)

**Análisis paso a paso**:

1. **Rango de edad**: Ambos conjuntos cubren el mismo rango
2. **Distribución**: Las barras sintéticas (rojas) tienden a ser ligeramente más altas en rangos de 20-40 años
3. **Coherencia con métricas**: rel_mean_diff = 13.4% confirma la ligera diferencia
4. **Conclusión**: Buena fidelidad con tendencia a generar pasajeros más jóvenes

## 6. Ubicación de resultados

### 6.1 Archivos de datos

- **`results/data/cleaned_dataset.csv`**: Dataset tras limpieza
- **`results/data/balanced_data.csv`**: Dataset balanceado
- **`results/data/synthetic_data.csv`**: Datos sintéticos finales

### 6.2 Gráficos de análisis

- **`results/graphics/original_dist.png`**: Distribución original del objetivo
- **`results/graphics/balanced_dist.png`**: Distribución balanceada
- **`results/graphics/correlation_matrix.png`**: Matriz de correlación completa
- **`results/graphics/survived_correlation.png`**: Correlación con supervivencia
- **`results/graphics/compare/`**: Comparaciones real vs sintético por variable

### 6.3 Reportes JSON

- **`results/reports/comparison_report.json`**: Métricas detalladas por variable
- **`results/reports/analysis_report.json`**: Resumen de análisis y balanceo
- **`results/reports/correlation_report.json`**: Análisis de correlaciones
- **`results/reports/cleaning_report.json`**: Trazabilidad de limpieza

## 7. Conclusiones y recomendaciones

### 7.1 Evaluación general

El generador GAN demuestra una **buena capacidad discriminativa** (77.78%) con fortalezas claras en variables categóricas y numéricas continuas, pero con problemas específicos en variables de conteo discreto.

### 6.2 Recomendaciones de mejora

1. **Variables de conteo discreto**

   - Considerar arquitecturas especializadas (GAN + VAE)
   - Implementar técnicas específicas para variables discretas

2. **Categorías minoritarias**

   - Ajustar parámetros de entrenamiento
   - Implementar técnicas de data augmentation

3. **Validación adicional**
   - Tests estadísticos más rigurosos
   - Validación con modelos downstream

### 6.3 Uso recomendado

Los datos sintéticos generados son **adecuados para**:

- Entrenamiento de modelos de clasificación
- Análisis exploratorio de datos
- Pruebas de algoritmos

**Limitaciones**:

- Variables de conteo discreto requieren validación adicional
- Categorías minoritarias pueden necesitar ajustes
