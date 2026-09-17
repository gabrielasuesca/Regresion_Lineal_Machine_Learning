# Regresión Lineal — Predicción de temperatura en Bogotá

Proyecto académico para la asignatura **Introducción a Machine Learning** de la Universidad de Cundinamarca.

## Objetivo

Estimar la temperatura del aire a partir de cinco variables meteorológicas:

- Humedad relativa
- Precipitación
- Presión atmosférica
- Velocidad del viento
- Radiación solar

La variable objetivo es `temperature_2m`.

## Fuente de datos

Los datos se obtuvieron mediante la **Open-Meteo Historical Weather API** para Bogotá (4.7110, -74.0721), con frecuencia horaria y periodo 2025-01-01 a 2025-12-31.

## Estructura

```text
Regresion_Lineal_Machine_Learning/
├── data/
│   └── datos_meteorologicos_bogota.csv
├── regresion_lineal_bogota.ipynb
├── regresion_lineal_bogota.py
├── requirements.txt
├── README.md
└── informe_regresion_lineal_bogota.pdf
```

## Resultados principales

- MAE: 0.8977 °C
- MSE: 1.2668
- RMSE: 1.1255 °C
- R²: 0.9015

## Validación

La prueba de Breusch-Pagan presentó un p-value extremadamente pequeño, por lo que se encontró evidencia de heterocedasticidad en los residuos.

Los VIF más altos fueron:

- Humedad relativa: 6.2959
- Radiación solar: 5.8676

Esto sugiere multicolinealidad entre algunos predictores y debe considerarse como limitación al interpretar los coeficientes.

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Después abrir `regresion_lineal_bogota.ipynb` en Jupyter o VS Code y ejecutar las celdas en orden.

## Referencia de la fuente

Open-Meteo Historical Weather API:
https://open-meteo.com/en/docs/historical-weather-api

## Autor

Gabriela Suesca Castillo
