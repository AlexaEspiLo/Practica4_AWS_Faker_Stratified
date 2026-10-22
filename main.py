
from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# PRACTICA 4: GENERACION DE DATOS SINTETICOS
# Y PARTICION ESTRATIFICADA
# ==========================================

# 1. CONFIGURACION DE FAKER Y SEMILLAS
fake = Faker("es_MX")
Faker.seed(42)
np.random.seed(42)

# 2. GENERACION DE 1000 REGISTROS SINTETICOS
registros = []

for i in range(1000):
    registros.append({
        "id_cliente": i + 1,
        "nombre": fake.name(),
        "email": fake.email(),

        # Campos adicionales de Faker
        "direccion": fake.address().replace("\n", ", "),
        "telefono": fake.phone_number(),

        "estado_civil": np.random.choice(
            ["Soltero", "Casado", "Divorciado"],
            p=[0.5, 0.4, 0.1]
        ),

        "ingreso_anual": round(
            np.random.uniform(100000, 1000000), 2
        ),

        "churn": np.random.choice(
            [0, 1],
            p=[0.8, 0.2]
        )
    })

df_sintetico = pd.DataFrame(registros)

print("\n=== DATASET SINTETICO GENERADO ===")
print(df_sintetico.head().to_string(index=False))

print("\n=== DIMENSIONES DEL DATASET ===")
print(f"Filas: {df_sintetico.shape[0]}")
print(f"Columnas: {df_sintetico.shape[1]}")

print("\n=== VALORES NULOS ===")
print(df_sintetico.isnull().sum())


# 3. SEPARACION DE VARIABLES X E y
# X = Variables predictoras
# y = Variable objetivo (churn)

# Excluimos los identificadores y datos de contacto
X = df_sintetico.drop(
    columns=[
        "id_cliente",
        "nombre",
        "email",
        "direccion",
        "telefono",
        "churn"
    ]
)

y = df_sintetico["churn"]

print("\n=== VARIABLES PREDICTORAS X ===")
print(X.head())

print("\n=== VARIABLE OBJETIVO y ===")
print(y.head())


# 4. CODIFICACION DE VARIABLES CATEGORICAS
# Transformar estado_civil en columnas numericas

X = pd.get_dummies(
    X,
    columns=["estado_civil"],
    dtype=int
)

print("\n=== VARIABLES CODIFICADAS ===")
print(X.head())

print("\nColumnas resultantes:")
print(X.columns.tolist())


# 5. PARTICION ESTRATIFICADA TRAIN-TEST
# 80% entrenamiento y 20% prueba

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== PARTICION ESTRATIFICADA ===")

print(f"Train X: {X_train.shape}")
print(f"Test X: {X_test.shape}")

print(f"Train y: {y_train.shape}")
print(f"Test y: {y_test.shape}")

print(f"\nChurn original: {y.mean():.2%}")
print(f"Churn Train: {y_train.mean():.2%}")
print(f"Churn Test: {y_test.mean():.2%}")


# 6. TABLA COMPARATIVA DE CHURN
resumen = pd.DataFrame({
    "Conjunto": [
        "Dataset completo",
        "Entrenamiento",
        "Prueba"
    ],
    "Registros": [
        len(y),
        len(y_train),
        len(y_test)
    ],
    "Churn_0": [
        (y == 0).sum(),
        (y_train == 0).sum(),
        (y_test == 0).sum()
    ],
    "Churn_1": [
        (y == 1).sum(),
        (y_train == 1).sum(),
        (y_test == 1).sum()
    ],
    "Porcentaje_Churn": [
        round(y.mean() * 100, 2),
        round(y_train.mean() * 100, 2),
        round(y_test.mean() * 100, 2)
    ]
})

print("\n=== COMPARACION DE PROPORCIONES ===")
print(resumen.to_string(index=False))


# 7. VALIDACION DE RESULTADOS
print("\n=== VALIDACION FINAL ===")

assert len(df_sintetico) == 1000
assert len(X_train) == 800
assert len(X_test) == 200
assert len(y_train) == 800
assert len(y_test) == 200

assert not X.isnull().any().any()
assert set(y.unique()) == {0, 1}

print("Dataset de 1000 registros verificado.")
print("Particion 80/20 correcta.")
print("Variables codificadas correctamente.")
print("Estratificacion aplicada exitosamente.")


# 8. EXPORTACION DE ARCHIVOS CSV
df_sintetico.to_csv(
    "dataset_sintetico.csv",
    index=False
)

df_train = X_train.copy()
df_train["churn"] = y_train

df_test = X_test.copy()
df_test["churn"] = y_test

df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)

resumen.to_csv(
    "comparacion_churn.csv",
    index=False
)

print("\n=== ARCHIVOS GENERADOS ===")
print("dataset_sintetico.csv")
print("train.csv")
print("test.csv")
print("comparacion_churn.csv")

print("\n=== PRACTICA 4 FINALIZADA ===")
