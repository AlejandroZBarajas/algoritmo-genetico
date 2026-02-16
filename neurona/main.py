import random
import math
import matplotlib.pyplot as plt

# =========================
# 1. Cargar dataset
# =========================
FILE_PATH = "./neurona/iris/iris.data"

dataset = []

with open(FILE_PATH, "r") as file:
    for line in file:
        line = line.strip()
        if line == "":
            continue

        parts = line.split(",")
        numeric_part = parts[:4]  # ignoramos la clase string

        row = [float(v) for v in numeric_part]
        dataset.append(row)

m = len(dataset)
n = len(dataset[0])

# =========================
# 2. Construir X y Y
# =========================
X = []
Y = []

for row in dataset:
    X.append([1.0, row[0], row[1], row[2]])  # bias
    Y.append([row[3]])

# =========================
# 3. Learning rates y épocas
# =========================
learning_rates = [0.9, 0.7, 0.5, 0.1, 0.3]
epochs = 30

weights_history = {}      # lr -> [ [w_epoch0], [w_epoch1], ... ]
error_norm_history = {}   # lr -> [ |E|_epoch0, |E|_epoch1, ... ]

# =========================
# 4. Entrenamiento
# =========================
for lr in learning_rates:

    # Pesos iniciales aleatorios [0,1]
    W = [[random.random()] for _ in range(n)]

    weights_history[lr] = []
    error_norm_history[lr] = []

    for epoch in range(epochs):

        # ---- Forward ----
        Yc = []
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += X[i][j] * W[j][0]
            Yc.append([s])

        # ---- Error ----
        E = []
        for i in range(m):
            E.append([Y[i][0] - Yc[i][0]])

        # ---- Norma del error ----
        norm_E = 0.0
        for i in range(m):
            norm_E += E[i][0] ** 2
        norm_E = math.sqrt(norm_E)
        error_norm_history[lr].append(norm_E)

        # ---- ΔW = Xᵀ · E ----
        deltaW = [[0.0] for _ in range(n)]
        for j in range(n):
            s = 0.0
            for i in range(m):
                s += X[i][j] * E[i][0]
            deltaW[j][0] = s

        # ---- Actualización ----
        for j in range(n):
            W[j][0] -= lr * deltaW[j][0]

        # Guardar pesos
        weights_history[lr].append([w[0] for w in W])

# =========================
# 5. Gráfica: Pesos vs épocas
# =========================
plt.figure()
for lr in learning_rates:
    for k in range(n):
        plt.plot(
            range(epochs),
            [w[k] for w in weights_history[lr]]
        )

plt.xlabel("Épocas")
plt.ylabel("Pesos W")
plt.title("Evolución de los pesos para distintos learning rates")
plt.savefig("pesos.png")
plt.close()
# =========================
# 6. Gráfica: Norma del error
# =========================
plt.figure()
for lr in learning_rates:
    plt.plot(
        range(epochs),
        error_norm_history[lr]
    )

plt.xlabel("Épocas")
plt.ylabel("|Error|")
plt.savefig("error.png")
plt.close()