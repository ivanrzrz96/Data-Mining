# Análisis de Salarios y Series Temporales Financieras

Proyecto de minería de datos sobre dos conjuntos de datos independientes: un dataset de 6.684 registros de empleados con información laboral y de formación, y la serie histórica de cotización de Microsoft entre 2015 y 2021.

El objetivo es doble. En el primer notebook se modelizan los determinantes del salario, se evalúa si el género es predecible a partir del perfil profesional y se segmenta la plantilla mediante clustering jerárquico. En el segundo se analiza la predictibilidad de una serie financiera diaria comparando tres enfoques clásicos frente a un baseline de persistencia.

## Contenido

| Notebook | Técnicas aplicadas |
|---|---|
| `01_salary_multivariate_analysis.ipynb` | Análisis exploratorio, tratamiento de nulos y outliers, regresión lineal (OLS), selección de variables forward, PCA, análisis factorial con rotación varimax, regresión logística con regularización LASSO y clustering jerárquico aglomerativo |
| `02_stock_price_forecasting.ipynb` | Contrastes de estacionariedad (ACF/PACF, Dickey-Fuller aumentado), descomposición de la serie, suavizado exponencial de Holt, modelo autorregresivo y optimización ARIMA |

## Hallazgos principales

- **La experiencia es el principal determinante del salario.** El modelo de regresión lineal explica el 74.5% de su variabilidad, y la selección forward confirma que las ocho variables retenidas son todas relevantes. El diagnóstico revela, sin embargo, colinealidad apreciable entre edad y experiencia (VIF de 9.08 y 13.11) que invierte el signo del coeficiente de la edad e impide su lectura individual.

- **La selección forward maximiza el R² a costa de la parsimonia.** El método retiene las ocho variables, pero la curva de rendimiento se estabiliza en la quinta: la ganancia entre ese punto y el conjunto completo es de 0.0015 puntos de R², un orden de magnitud por debajo de la desviación típica de la validación cruzada. Criterios como el AIC o el R² ajustado habrían seleccionado un modelo más simple con ajuste equivalente.

- **El PCA reduce la multicolinealidad pero no compensa.** Baja el Condition Number de 21.5 a 2.32, ya que las componentes son ortogonales por construcción, pero pierde más de cinco puntos de R² (0.6925 frente a 0.7454) y toda la interpretabilidad. Con un KMO de 0.3987 y seis componentes necesarias para alcanzar el 95% de varianza, la reducción de dimensionalidad efectiva es mínima.

- **El género no es predecible a partir de las variables laborales disponibles.** La regresión logística con LASSO alcanza un AUC de 0.6295 y un accuracy del 58%, apenas tres puntos por encima del 54.9% que se obtendría prediciendo siempre la clase mayoritaria. El recall es marcadamente asimétrico (0.35 frente a 0.78), señal de que el modelo se refugia en la clase dominante.

- **El clustering jerárquico con enlace de Ward segmenta la plantilla en tres grupos por seniority**: perfiles junior (2.343 empleados, 70.614 USD de salario medio), núcleo técnico consolidado (3.547, 132.837 USD) y perfiles senior o de gestión (794, 168.881 USD). Los tres se ordenan de forma consistente en edad, experiencia y salario, y el grupo mejor retribuido concentra un 67.0% de hombres frente a proporciones próximas al equilibrio en los otros dos. Los coeficientes de silueta próximos a cero indican, no obstante, que los grupos se solapan: la segmentación debe leerse como una discretización razonada del continuo de seniority, no como el hallazgo de segmentos naturales bien delimitados.

- **La serie de cotización se comporta esencialmente como un paseo aleatorio.** Ninguno de los cuatro candidatos evaluados mejora de forma apreciable al baseline de persistencia: el mejor de ellos, un ARIMA(1,1,0) sin deriva, reduce el RMSE en apenas un 1.1% sobre un horizonte de 30 días. La heterocedasticidad detectada en los residuos indica que la estructura aprovechable está en la volatilidad, no en el nivel del precio.

- **Un término no significativo puede dominar el error de predicción.** El intercepto que `auto_arima` incorpora al modelo (p = 0.139) proyecta una deriva ascendente que la serie real no siguió, y eleva el RMSE de 10.665 a 15.370. Es un recordatorio de que el criterio AIC premia mejoras dentro de la muestra que no se sostienen fuera de ella.

## Estructura del repositorio

    data/
        Salary_MD.csv                              # salarios con nulos y outliers
        Salary.csv                                 # salarios sin depurar (clustering)
        Microsoft_Stock.csv                        # cotización diaria 2015-2021
    01_salary_multivariate_analysis.ipynb
    02_stock_price_forecasting.ipynb
    requirements.txt

## Ejecución

```bash
pip install -r requirements.txt
jupyter notebook
```

Los notebooks incluyen los resultados de ejecución guardados, por lo que pueden consultarse directamente sin necesidad de ejecutarlos. Las rutas de lectura son relativas a la raíz del repositorio.

## Decisiones metodológicas

- **Imputación de nulos en `Gender`.** El salario medio de los registros sin género informado (145.388 USD) difiere de forma apreciable del resto (116.123 USD), lo que descarta una ausencia aleatoria. Se imputa mediante la moda segmentada por puesto de trabajo, apoyándose en la asociación detectada por la V de Cramér, en lugar de aplicar la moda global, que reforzaría artificialmente la clase mayoritaria.

- **Eliminación de `Race`.** Su V de Cramér con `Gender`, calculada con corrección de sesgo, es 0.0: la asociación observada no supera la esperable por azar dado el número de categorías, por lo que no aportaba información al modelo de clasificación.

- **Tratamiento de outliers.** Se emplea el criterio del rango intercuartílico en las tres variables continuas por su robustez frente a los propios valores extremos, aplicando un criterio homogéneo que permite comparar la atipicidad entre dimensiones.

- **Particiones aleatorias en validación cruzada.** Los registros del fichero no están en orden aleatorio, sino agrupados por perfiles similares. La partición secuencial que aplica `KFold` por defecto producía folds con R² fuertemente negativo, un artefacto del orden de los datos que sobreestimaba la varianza del modelo. Con particiones aleatorias la desviación típica del MSE cae al 4.8% de su media, confirmando que el modelo es estable.

- **Número de clústeres.** Se determina a partir del mayor salto en la distancia de fusión del dendrograma (18.19 al pasar de 3 a 2 grupos) y se contrasta con el coeficiente de silueta para k entre 2 y 6. Ambos criterios no coinciden —la silueta favorece k = 2— y la discrepancia se documenta explícitamente en lugar de resolverse en silencio.

- **Selección de órdenes del ARIMA.** La búsqueda se realiza empleando únicamente el conjunto de entrenamiento, de forma que los 30 días reservados para validación no influyan en la elección del modelo.

- **Baseline de persistencia.** La comparativa de series temporales incluye la predicción naive (el precio se mantiene en el último valor conocido) como referencia obligada: un modelo que no la supera no aporta valor predictivo, por muy correcta que sea su especificación estadística.

## Limitaciones

- El escalado de las variables continuas se realiza sobre el conjunto completo antes de separar entrenamiento y prueba. Se incluye una celda de control que repite el ajuste escalando únicamente con el conjunto de entrenamiento y verifica que las conclusiones no cambian.

- En el clustering se mantienen las 129 categorías originales de `Job Title` codificadas con one-hot encoding, lo que genera un espacio disperso en el que la distancia euclídea pierde capacidad de discriminación y explica en parte los coeficientes de silueta bajos. Es una decisión deliberada para poder interpretar los puestos reales predominantes en cada grupo.

- Los residuos no siguen una distribución normal en ninguno de los modelos, algo esperable con datos salariales y financieros. Con el tamaño de muestra disponible las estimaciones puntuales siguen siendo fiables, pero los intervalos de confianza deben interpretarse con cautela.

- El análisis factorial presenta un caso Heywood en una de las variables (comunalidad superior a 1, unicidad negativa), por lo que su estimación no es fiable y se señala explícitamente en la interpretación.

- El modelo AR(1) deja autocorrelación residual significativa en los rezagos 10 y 20 (p = 0.035 y p = 0.049), lo que sugiere estructura temporal sin modelizar a medio plazo.

## Stack

Python, pandas, NumPy, scikit-learn, statsmodels, factor-analyzer, mlxtend, pmdarima, SciPy, matplotlib, seaborn.
