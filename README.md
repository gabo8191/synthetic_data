# Proyecto: Generación de datos sintéticos (Titanic)

## Estructura del proyecto

```text
data/
  Titanic-Dataset.csv                    # Dataset original
results/
  data/                                  # Datasets procesados
    cleaned_dataset.csv                  # Dataset limpio
    balanced_data.csv                    # Dataset balanceado
    synthetic_data.csv                   # Datos sintéticos generados
  graphics/                              # Gráficos y visualizaciones
    original_dist.png                    # Distribución original
    balanced_dist.png                    # Distribución balanceada
    correlation_matrix.png               # Matriz de correlación
    survived_correlation.png             # Correlación con supervivencia
    compare/                             # Comparaciones real vs sintético
      compare_*.png                      # Gráficos por variable
  reports/                               # Reportes JSON
    cleaning_report.json                 # Reporte de limpieza
    analysis_report.json                 # Reporte de análisis y balanceo
    comparison_report.json               # Reporte de comparación
    correlation_report.json              # Reporte de correlación
    pipeline.log                         # Log del pipeline
docs/
  algorithm.md                           # Explicación del pipeline y GAN
  results_analysis.md                    # Interpretación de resultados
  glossary.md                            # Glosario técnico
  variables_explanation.md               # Explicación de variables
src/                                     # Código fuente
  config.py                              # Configuración centralizada
  utils.py                               # Funciones utilitarias
  validation.py                          # Validaciones
  data/loader.py                         # Carga y limpieza
  analysis/                              # Análisis de datos
  models/                                # Modelos GAN
  pipeline/main.py                       # Pipeline principal
main.py                                  # Punto de entrada
requirements.txt                         # Dependencias
```

## Uso

### 1. Configuración del entorno

```bash
# Crear entorno virtual
python -m venv env_sintetic

# Activar entorno (Windows PowerShell)
env_sintetic\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecución

```bash
python main.py
```

### 3. Resultados

Los resultados se generan en la carpeta `results/`:

- **Datos**: `results/data/` - Datasets procesados
- **Gráficos**: `results/graphics/` - Visualizaciones
- **Reportes**: `results/reports/` - Análisis detallados

## Documentación

- **`docs/algorithm.md`**: Explicación técnica del pipeline y arquitectura GAN
- **`docs/results_analysis.md`**: Interpretación detallada de resultados y métricas
- **`docs/glossary.md`**: Definiciones de términos técnicos
- **`docs/variables_explanation.md`**: Explicación de cada variable del dataset

## Dataset fuente

Dataset original de la competencia Titanic de Kaggle: [Datos de la competencia Titanic](https://www.kaggle.com/competitions/titanic/data)
