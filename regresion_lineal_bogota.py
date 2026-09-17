# Regresión lineal — Predicción de temperatura en Bogotá
# Gabriela Suesca Castillo
# Universidad de Cundinamarca — Introducción a Machine Learning
#
# Fuente de datos: Open-Meteo Historical Weather API
# Periodo: 2025-01-01 a 2025-12-31

# --- Celda 1 ---
import requests
import pandas as pd
import os

# --- Celda 2 ---
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 4.7110,
    "longitude": -74.0721,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "shortwave_radiation"
    ]),
    "timezone": "America/Bogota"
}

response = requests.get(url, params=params)
response.raise_for_status()

datos = response.json()

print("Datos descargados correctamente")

# --- Celda 3 ---
df = pd.DataFrame(datos["hourly"])

df.head()

# --- Celda 4 ---
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# --- Celda 5 ---
os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/datos_meteorologicos_bogota.csv",
    index=False
)

print("CSV guardado correctamente")

# --- Celda 6 ---
df.info()

# --- Celda 7 ---
df.describe()

# --- Celda 8 ---
df.isnull().sum()

# --- Celda 9 ---
df.duplicated().sum()

# --- Celda 10 ---
df.describe()

# --- Celda 11 ---
df.corr(numeric_only=True)

# --- Celda 12 ---
import matplotlib.pyplot as plt
import seaborn as sns

print("Librerías listas")

# --- Celda 13 ---
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriz de correlación")
plt.tight_layout()
plt.show()

# --- Celda 14 ---
plt.figure(figsize=(8, 5))

plt.hist(
    df["temperature_2m"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribución de la temperatura")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# --- Celda 15 ---
plt.figure(figsize=(8, 5))

plt.scatter(
    df["relative_humidity_2m"],
    df["temperature_2m"],
    alpha=0.4
)

plt.title("Temperatura vs. humedad relativa")
plt.xlabel("Humedad relativa (%)")
plt.ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# --- Celda 16 ---
plt.figure(figsize=(8, 5))

plt.scatter(
    df["wind_speed_10m"],
    df["temperature_2m"],
    alpha=0.4
)

plt.title("Temperatura vs. velocidad del viento")
plt.xlabel("Velocidad del viento")
plt.ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# --- Celda 17 ---
plt.figure(figsize=(8, 5))

plt.scatter(
    df["surface_pressure"],
    df["temperature_2m"],
    alpha=0.4
)

plt.title("Temperatura vs. presión atmosférica")
plt.xlabel("Presión atmosférica")
plt.ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# --- Celda 18 ---
plt.figure(figsize=(8, 5))

plt.scatter(
    df["precipitation"],
    df["temperature_2m"],
    alpha=0.4
)

plt.title("Temperatura vs. precipitación")
plt.xlabel("Precipitación")
plt.ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# --- Celda 19 ---
plt.figure(figsize=(8, 5))

plt.scatter(
    df["shortwave_radiation"],
    df["temperature_2m"],
    alpha=0.4
)

plt.title("Temperatura vs. radiación solar")
plt.xlabel("Radiación solar")
plt.ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()

# --- Celda 20 ---
df.describe()

# --- Celda 21 ---
# Variables del modelo

X = df[
    [
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "shortwave_radiation"
    ]
]

y = df["temperature_2m"]

print("Variables predictoras:", X.columns.tolist())
print("Variable objetivo:", y.name)

# --- Celda 22 ---
df = df.sort_values("time").reset_index(drop=True)

# --- Celda 23 ---
# Separación temporal 80/20

corte = int(len(df) * 0.80)

X = df[
    [
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "shortwave_radiation"
    ]
]

y = df["temperature_2m"]

X_train = X.iloc[:corte]
X_test = X.iloc[corte:]

y_train = y.iloc[:corte]
y_test = y.iloc[corte:]

print("Datos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)

# --- Celda 24 ---
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

# --- Celda 25 ---
coeficientes = pd.DataFrame({
    "Variable": X_train.columns,
    "Coeficiente": modelo.coef_
})

coeficientes

# --- Celda 26 ---
print("Intercepto:", modelo.intercept_)

# --- Celda 27 ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = mse ** 0.5
r2 = r2_score(y_test, predicciones)

print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

# --- Celda 28 ---
comparacion = pd.DataFrame({
    "Real": y_test.values,
    "Predicha": predicciones
})

comparacion.head(10)

# --- Celda 29 ---
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predicciones, alpha=0.5)

minimo = min(y_test.min(), predicciones.min())
maximo = max(y_test.max(), predicciones.max())

plt.plot([minimo, maximo], [minimo, maximo], linestyle="--")

plt.title("Temperatura real vs. temperatura predicha")
plt.xlabel("Temperatura real (°C)")
plt.ylabel("Temperatura predicha (°C)")

plt.tight_layout()
plt.show()

# --- Celda 30 ---
residuos = y_test - predicciones

# --- Celda 31 ---
plt.figure(figsize=(8, 5))

plt.scatter(predicciones, residuos, alpha=0.5)
plt.axhline(0, linestyle="--")

plt.title("Residuos vs. valores predichos")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")

plt.tight_layout()
plt.show()

# --- Celda 32 ---
plt.figure(figsize=(8, 5))

plt.hist(residuos, bins=30, edgecolor="black")

plt.title("Distribución de los residuos")
plt.xlabel("Residuo")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# --- Celda 33 ---
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(residuos, line="45", fit=True)

plt.title("Gráfico Q-Q de los residuos")
plt.tight_layout()
plt.show()

# --- Celda 34 ---
from statsmodels.stats.diagnostic import het_breuschpagan

X_test_const = sm.add_constant(X_test)

resultado_bp = het_breuschpagan(
    residuos,
    X_test_const
)

labels = [
    "LM Statistic",
    "LM-Test p-value",
    "F-Statistic",
    "F-Test p-value"
]

dict(zip(labels, resultado_bp))

# --- Celda 35 ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(X)

vif = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})

vif

# --- Celda 36 ---
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predicciones, alpha=0.4)

minimo = min(y_test.min(), predicciones.min())
maximo = max(y_test.max(), predicciones.max())

plt.plot(
    [minimo, maximo],
    [minimo, maximo],
    linestyle="--"
)

plt.title("Valores reales vs. predichos")
plt.xlabel("Temperatura real (°C)")
plt.ylabel("Temperatura predicha (°C)")

plt.tight_layout()
plt.show()
