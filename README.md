## Proyecto: Generación de datos sintéticos (Titanic)

Estructura final:

```
data/
  Titanic-Dataset.csv
results/
  data/
  graphics/
  reports/
docs/
  algorithm.md          # Explicación del pipeline y GAN
  results_analysis.md   # Interpretación de resultados y métricas
  glossary.md           # Glosario técnico y definiciones
src/
  algorithm.py        # GAN simple para generar datos
  analyzer.py         # Análisis, balanceo y comparación
  loader.py           # Carga y limpieza del dataset
  main.py             # Pipeline principal (lee data, limpia, balancea, sintetiza)
main.py               # entrypoint que llama a src.pipeline.main
requirements.txt
```

### Uso

1. Crear y activar entorno

```bash
python -m venv env_sintetic
env_sintetic\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

2. Ejecutar

```bash
python main.py
```

Resultados en `results/`.

### Dataset fuente

El dataset original proviene de la competencia Titanic de Kaggle: [Datos de la competencia Titanic](https://www.kaggle.com/competitions/titanic/data).

### Documentación

- Ver `docs/algorithm.md` para entender el flujo del pipeline, decisiones de preprocesamiento, el entrenamiento del generador y la función unificada de gráficos.
- Ver `docs/results_analysis.md` para interpretar `results/reports/*.json` y las figuras de `results/graphics/`.
- Ver `docs/glossary.md` para definiciones formales (p. ej., GAN, estratificación, bins, fidelidad, drift).

### Objetivo y organización

- `data/`: datasets de entrada (p.ej. Titanic)
- `results/`: resultados generados por el pipeline
  - `results/data`: datasets limpios, balanceados y sintéticos
  - `results/graphics`: gráficos de distribución y comparativas
  - `results/reports`: reportes JSON de limpieza y análisis
- `docs/`: documentación técnica y guía de interpretación de resultados
- `src/`: código fuente organizado por responsabilidad
