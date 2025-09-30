# Guía de Reportes JSON - Variables y Estructuras

Este documento explica detalladamente todas las variables y estructuras que se encuentran en los reportes JSON generados por el pipeline de generación de datos sintéticos.

## 1. cleaning_report.json

### 1.1 Estructura general

```json
{
    "load_info": { ... },
    "duplicates": { ... },
    "final_info": { ... },
    "quality_report": { ... }
}
```

### 1.2 load_info - Información de carga

```json
"load_info": {
    "file": "Titanic-Dataset.csv",           // Nombre del archivo original
    "original_shape": [891, 12],             // [filas, columnas] del dataset original
    "original_columns": [                    // Lista de columnas originales
        "PassengerId", "Survived", "Pclass", "Name", "Sex",
        "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ]
}
```

### 1.3 duplicates - Información de duplicados

```json
"duplicates": {
    "before": 0,        // Número de duplicados antes de la limpieza
    "removed": 0        // Número de duplicados removidos
}
```

### 1.4 final_info - Información final del dataset

```json
"final_info": {
    "final_shape": [891, 12],               // [filas, columnas] después de limpieza
    "missing_values": {                     // Valores faltantes por columna
        "passengerid": 0, "survived": 0, "pclass": 0, "name": 0,
        "sex": 0, "age": 0, "sibsp": 0, "parch": 0,
        "ticket": 0, "fare": 0, "cabin": 0, "embarked": 0
    },
    "data_types": {                         // Tipos de datos por columna
        "passengerid": "float64", "survived": "float64", "pclass": "float64",
        "name": "object", "sex": "object", "age": "float64",
        "sibsp": "float64", "parch": "float64", "ticket": "object",
        "fare": "float64", "cabin": "object", "embarked": "object"
    }
}
```

### 1.5 quality_report - Reporte de calidad

```json
"quality_report": {
    "shape": [891, 12],                     // Dimensiones finales
    "missing_values": { ... },              // Valores faltantes (igual que final_info)
    "missing_percentage": {                 // Porcentaje de valores faltantes por columna
        "passengerid": 0.0, "survived": 0.0, "pclass": 0.0, "name": 0.0,
        "sex": 0.0, "age": 0.0, "sibsp": 0.0, "parch": 0.0,
        "ticket": 0.0, "fare": 0.0, "cabin": 0.0, "embarked": 0.0
    },
    "duplicates": 0,                        // Número total de duplicados
    "memory_usage_mb": 0.29467201232910156, // Uso de memoria en MB
    "data_types": { ... },                  // Tipos de datos (igual que final_info)
    "numeric_columns": [                    // Lista de columnas numéricas
        "passengerid", "survived", "pclass", "age", "sibsp", "parch", "fare"
    ],
    "categorical_columns": [                // Lista de columnas categóricas
        "name", "sex", "ticket", "cabin", "embarked"
    ]
}
```

## 2. analysis_report.json

### 2.1 Estructura general

```json
{
    "class_analysis": { ... },
    "balancing": { ... },
    "validation": { ... }
}
```

### 2.2 class_analysis - Análisis de clases

```json
"class_analysis": {
    "original_distribution": {              // Distribución original de clases
        "0.0": 439,                        // 439 pasajeros no sobrevivieron
        "1.0": 273                         // 273 pasajeros sobrevivieron
    },
    "minority_class": 1.0,                 // Clase minoritaria (sobrevivientes)
    "imbalance_ratio": 0.621867881548975   // Ratio de desbalance (minoría/mayoría)
}
```

### 2.3 balancing - Información de balanceo

```json
"balancing": {
    "method": "RandomUnderSampler",         // Método de balanceo utilizado
    "new_distribution": {                   // Nueva distribución después del balanceo
        "0.0": 273,                        // 273 pasajeros no sobrevivieron
        "1.0": 273                         // 273 pasajeros sobrevivieron
    },
    "samples_removed": 166,                 // Muestras removidas para balancear
    "original_samples": 712,                // Muestras originales (80% del dataset)
    "balanced_samples": 546                 // Muestras después del balanceo
}
```

### 2.4 validation - Validación del balanceo

```json
"validation": {
    "original_imbalance_ratio": 1.6080586080586081,  // Ratio de desbalance original
    "balanced_imbalance_ratio": 1.0,                 // Ratio después del balanceo (1.0 = perfecto)
    "improvement_factor": 1.6080586080586081,        // Factor de mejora
    "is_balanced": true                              // Si el dataset está balanceado
}
```

## 3. comparison_report.json

### 3.1 Estructura general

```json
{
    "pclass": { ... },      // Variable numérica
    "age": { ... },         // Variable numérica
    "sibsp": { ... },       // Variable numérica
    "parch": { ... },       // Variable numérica
    "fare": { ... },        // Variable numérica
    "sex": { ... },         // Variable categórica
    "embarked": { ... },    // Variable categórica
    "real_freq": { ... },   // Frecuencias reales del objetivo
    "synthetic_freq": { ... }, // Frecuencias sintéticas del objetivo
    "_metadata": { ... }    // Metadatos del reporte
}
```

### 3.2 Variables numéricas - Estructura

```json
"pclass": {
    "type": "numeric",                      // Tipo de variable
    "real_mean": 2.308988764044944,         // Media de datos reales
    "synthetic_mean": 2.130239343523979,    // Media de datos sintéticos
    "real_std": 0.8335632357216292,         // Desviación estándar real
    "synthetic_std": 0.8523226236973886,    // Desviación estándar sintética
    "rel_mean_diff": 0.07741459054728417,   // Diferencia relativa de la media
    "std_rel_diff": 0.022505056811339796,   // Diferencia relativa de la desviación estándar
    "high_drift": false                     // Si tiene drift alto (true/false)
}
```

**Explicación de métricas numéricas**:

- **`real_mean`**: Promedio de los datos reales
- **`synthetic_mean`**: Promedio de los datos sintéticos
- **`real_std`**: Desviación estándar de los datos reales
- **`synthetic_std`**: Desviación estándar de los datos sintéticos
- **`rel_mean_diff`**: `|media_sintética - media_real| / (|media_real| + 1e-8)`
- **`std_rel_diff`**: `|std_sintética - std_real| / std_real`
- **`high_drift`**: `true` si `rel_mean_diff > 0.3`

### 3.3 Variables categóricas - Estructura

```json
"sex": {
    "type": "categorical",                  // Tipo de variable
    "categories": ["female", "male"],       // Lista de categorías
    "real_freq": {                          // Frecuencias reales (proporciones)
        "male": 0.6446629213483146,         // 64.47% hombres
        "female": 0.3553370786516854        // 35.53% mujeres
    },
    "synthetic_freq": {                     // Frecuencias sintéticas (proporciones)
        "female": 0.3523790240287781,       // 35.24% mujeres
        "male": 0.648088276386261           // 64.81% hombres
    },
    "max_difference": 0.003425355037946387, // Diferencia máxima entre categorías
    "high_drift": false                     // Si tiene drift alto (true/false)
}
```

**Explicación de métricas categóricas**:

- **`categories`**: Lista de todas las categorías únicas
- **`real_freq`**: Proporción de cada categoría en datos reales (suma = 1.0)
- **`synthetic_freq`**: Proporción de cada categoría en datos sintéticos (suma = 1.0)
- **`max_difference`**: Máxima diferencia absoluta entre frecuencias reales y sintéticas
- **`high_drift`**: `true` si `max_difference > 0.2`

### 3.4 Variable objetivo - Estructura

```json
"real_freq": {                              // Frecuencias reales del objetivo
    "0.0": 0.6165730337078652,              // 61.66% no sobrevivieron
    "1.0": 0.38342696629213485              // 38.34% sobrevivieron
},
"synthetic_freq": {                         // Frecuencias sintéticas del objetivo
    "0.0": 0.604,                           // 60.4% no sobrevivieron
    "1.0": 0.396                            // 39.6% sobrevivieron
}
```

### 3.5 Metadatos del reporte

```json
"_metadata": {
    "total_variables": 9,                   // Total de variables evaluadas
    "high_drift_variables": 2,              // Variables con drift alto
    "discrimination_percentage": 77.78,     // Porcentaje discriminador
    "quality_level": "Bueno"                // Nivel de calidad general
}
```

**Explicación de metadatos**:

- **`total_variables`**: Número total de variables comparadas
- **`high_drift_variables`**: Variables que superan el umbral de drift
- **`discrimination_percentage`**: `((total_variables - high_drift_variables) / total_variables) * 100`
- **`quality_level`**: Clasificación de calidad basada en el porcentaje discriminador

## 4. correlation_report.json

### 4.1 Estructura general

```json
{
    "correlation_matrix": { ... },          // Matriz de correlación completa
    "survived_correlations": { ... },       // Correlaciones con supervivencia
    "strong_correlations": { ... },         // Correlaciones fuertes identificadas
    "label_encoders_mapping": { ... },      // Mapeo de encoders categóricos
    "patterns_analysis": { ... }            // Análisis de patrones
}
```

### 4.2 correlation_matrix - Matriz de correlación

```json
"correlation_matrix": {
    "passengerid": {                        // Correlaciones de passengerid con todas las variables
        "passengerid": 1.0,                 // Correlación consigo misma (siempre 1.0)
        "survived": -0.0050066607670665175, // Correlación con supervivencia
        "pclass": -0.03514399403038102,     // Correlación con clase
        // ... más correlaciones
    },
    "survived": {                           // Correlaciones de survived con todas las variables
        "passengerid": -0.0050066607670665175,
        "survived": 1.0,                    // Correlación consigo misma
        "pclass": -0.33848103596101514,     // Correlación fuerte negativa con clase
        "sex": -0.543351380657755,          // Correlación fuerte negativa con sexo
        // ... más correlaciones
    }
    // ... más variables
}
```

**Interpretación de correlaciones**:

- **Valores cercanos a 1.0**: Correlación positiva fuerte
- **Valores cercanos a -1.0**: Correlación negativa fuerte
- **Valores cercanos a 0.0**: Sin correlación
- **Umbral de correlación fuerte**: |r| > 0.5

### 4.3 survived_correlations - Correlaciones con supervivencia

```json
"survived_correlations": {
    "sex": -0.543351380657755,              // Correlación más fuerte con supervivencia
    "pclass": -0.33848103596101514,         // Segunda correlación más fuerte
    "fare": 0.2573065223849626,             // Correlación positiva moderada
    // ... más correlaciones ordenadas por magnitud
}
```

### 4.4 strong_correlations - Correlaciones fuertes

```json
"strong_correlations": [
    {
        "variable1": "pclass",              // Primera variable
        "variable2": "fare",                // Segunda variable
        "correlation": -0.5494996199439076, // Valor de correlación
        "strength": "strong_negative"       // Tipo de correlación
    }
    // ... más correlaciones fuertes
]
```

### 4.5 label_encoders_mapping - Mapeo de encoders

```json
"label_encoders_mapping": {
    "sex": {
        "classes": ["female", "male"],      // Categorías originales
        "encoded_values": [0, 1]            // Valores codificados
    },
    "embarked": {
        "classes": ["C", "Q", "S"],         // Categorías originales
        "encoded_values": [0, 1, 2]         // Valores codificados
    }
}
```

### 4.6 patterns_analysis - Análisis de patrones

```json
"patterns_analysis": {
    "strongest_positive": {                 // Correlación positiva más fuerte
        "variables": ["var1", "var2"],
        "correlation": 0.85
    },
    "strongest_negative": {                 // Correlación negativa más fuerte
        "variables": ["var1", "var2"],
        "correlation": -0.75
    },
    "most_correlated_with_target": {        // Variable más correlacionada con el objetivo
        "variable": "sex",
        "correlation": -0.543351380657755
    }
}
```

## 5. pipeline.log

### 5.1 Estructura del log

El archivo `pipeline.log` contiene el registro completo de la ejecución del pipeline, incluyendo:

- **Timestamps**: Fecha y hora de cada evento
- **Niveles de log**: INFO, WARNING, ERROR
- **Módulos**: Identificación del módulo que genera el log
- **Mensajes**: Descripción detallada de cada operación

### 5.2 Ejemplo de entrada de log

```
2025-09-29 19:41:33 | INFO | pipeline | === INICIANDO PIPELINE DE GENERACIÓN DE DATOS SINTÉTICOS ===
2025-09-29 19:41:33 | INFO | pipeline | Cargando datos...
2025-09-29 19:41:33 | INFO | pipeline | Dataset limpio: 891 filas, 12 columnas
2025-09-29 19:41:35 | INFO | gan | Iniciando generación GAN: 1000 muestras, 1400 épocas
2025-09-29 19:42:19 | INFO | comparison | Comparación completada. Porcentaje discriminador: 77.78%
```

## 6. Interpretación de umbrales y métricas

### 6.1 Umbrales de drift

- **Variables numéricas**: `rel_mean_diff > 0.3` = drift alto
- **Variables categóricas**: `max_difference > 0.2` = drift alto

### 6.2 Niveles de calidad

- **> 80%**: Excelente capacidad discriminativa
- **60-80%**: Buena capacidad discriminativa
- **< 60%**: Capacidad discriminativa limitada

### 6.3 Correlaciones

- **|r| > 0.7**: Correlación muy fuerte
- **0.5 < |r| ≤ 0.7**: Correlación fuerte
- **0.3 < |r| ≤ 0.5**: Correlación moderada
- **|r| ≤ 0.3**: Correlación débil

## 7. Uso de los reportes

### 7.1 Para análisis de calidad

- **`comparison_report.json`**: Evaluar la fidelidad de datos sintéticos
- **`analysis_report.json`**: Verificar el balanceo de clases
- **`cleaning_report.json`**: Validar la calidad de datos de entrada

### 7.2 Para análisis de correlaciones

- **`correlation_report.json`**: Entender relaciones entre variables
- **`survived_correlations`**: Identificar variables más importantes para supervivencia

### 7.3 Para debugging

- **`pipeline.log`**: Rastrear errores y validar el flujo de ejecución
- **Metadatos**: Verificar configuraciones y parámetros utilizados
