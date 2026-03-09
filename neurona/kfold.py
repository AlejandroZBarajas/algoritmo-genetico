import random
import math
import matplotlib.pyplot as plt
import os

# =====================================================================
# CONSTANTES
# =====================================================================
K = 3               # Número de folds
GENERACIONES = 30
ACTIVACION = "identidad"
PATH = "./algoritmo-genetico/neurona/"

# =====================================================================
# CARGA Y PREPROCESAMIENTO
# =====================================================================

def cargar_data():
    path = "./algoritmo-genetico/neurona/C233435.csv"
    dataset = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                row = [float(v) for v in parts]
                dataset.append(row)
            except ValueError:
                continue
    return dataset

def calcular_XyY(dataset):
    y_raw = []
    X_sin_bias = []
    for row in dataset:
        y_raw.append(row[-1])
        X_sin_bias.append(row[:-1])

    X_norm = normalizar_minmax(X_sin_bias)

    X_final = []
    for row in X_norm:
        X_final.append([1] + row)

    y_min = min(y_raw)
    y_max = max(y_raw)
    y_norm = []
    for v in y_raw:
        if y_max != y_min:
            y_norm.append((v - y_min) / (y_max - y_min))
        else:
            y_norm.append(0.0)

    return X_final, y_norm

def normalizar_minmax(X):
    num_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(num_features)]
    maxs = [max(row[j] for row in X) for j in range(num_features)]
    X_norm = []
    for row in X:
        norm_row = []
        for j in range(num_features):
            if maxs[j] == mins[j]:
                norm_row.append(0.0)
            else:
                norm_row.append((row[j] - mins[j]) / (maxs[j] - mins[j]))
        X_norm.append(norm_row)
    return X_norm

# =====================================================================
# PARTICIÓN EN FOLDS
# =====================================================================

def crear_folds(X, Y, k):
    """
    Divide X e Y en k folds lo más iguales posible.
    Los residuos se reparten en orden (fold 0, fold 1, ...).
    Ej: 152 filas, k=3 → [51, 51, 50]
    """
    n = len(X)
    base = n // k
    resto = n % k

    folds_X = []
    folds_Y = []
    inicio = 0

    for i in range(k):
        tam = base + (1 if i < resto else 0)
        folds_X.append(X[inicio:inicio + tam])
        folds_Y.append(Y[inicio:inicio + tam])
        inicio += tam

    # Mostrar distribución
    print("\n" + "=" * 50)
    print(f"DISTRIBUCIÓN DE FOLDS (K={k}, N={n})")
    print("=" * 50)
    letras = "ABCDEFGHIJ"
    for i in range(k):
        print(f"  Fold {letras[i]}: {len(folds_X[i])} observaciones")
    print("=" * 50)

    return folds_X, folds_Y

# =====================================================================
# NEURONA
# =====================================================================

def cargar_pesos(n):
    pesos = [round(random.uniform(-1, 1), 2) for _ in range(n)]
    return pesos

def activar_funcion(u, activacion):
    if activacion == "identidad":
        return u
    elif activacion == "escalon":
        return 1 if u >= 0 else 0
    elif activacion == "sigmoide":
        return 1 / (1 + math.exp(-u))
    elif activacion == "relu":
        return max(0.0, u)
    elif activacion == "tanh":
        return math.tanh(u)
    elif activacion == "leaky_relu":
        return u if u >= 0 else 0.01 * u

def calcular_Yc(X, W, activacion):
    Yc = []
    for fila in X:
        u = sum(fila[j] * W[j] for j in range(len(W)))
        Yc.append(activar_funcion(u, activacion))
    return Yc

def calcular_error_norma(Y, Yc):
    return math.sqrt(sum((yc - y) ** 2 for y, yc in zip(Y, Yc)))

def calcular_E(Y, Yc):
    return [yc - y for y, yc in zip(Y, Yc)]

def calcular_delta_w(matriz_x, errores, lr):
    n_pesos = len(matriz_x[0])
    delta_w = []
    for j in range(n_pesos):
        suma = sum(matriz_x[i][j] * errores[i] for i in range(len(errores)))
        delta_w.append(lr * suma / len(errores))
    return delta_w

def actualizar_pesos(w, delta_w):
    return [w[i] - delta_w[i] for i in range(len(w))]

def cargar_lr():
    return [0.01, 0.02, 0.03, 0.04, 0.05]

# =====================================================================
# ENTRENAMIENTO CON EARLY STOPPING
# =====================================================================

def entrenar_con_early_stopping(X_train, Y_train, X_val, Y_val, w_inicial, lr, generaciones, activacion):
    """
    Entrena por 'generaciones' épocas.
    En cada época calcula error_train y error_val.
    Si error_val sube respecto a la época anterior → EARLY STOP.
    Guarda los pesos de la época con menor error_val.
    
    Retorna:
        mejores_pesos, mejor_error_train, mejor_error_val, epoca_stop, hist_train, hist_val
    """
    w = w_inicial.copy()

    n_train = len(X_train)
    n_val   = len(X_val)

    hist_train = []
    hist_val   = []

    mejor_error_val   = float("inf")
    mejor_error_train = float("inf")
    mejores_pesos     = w.copy()
    epoca_stop        = generaciones  # por defecto completa

    error_val_anterior = float("inf")

    for g in range(generaciones):
        # --- Entrenamiento ---
        Yc_train = calcular_Yc(X_train, w, activacion)
        errores  = calcular_E(Y_train, Yc_train)
        delta_w  = calcular_delta_w(X_train, errores, lr)
        w        = actualizar_pesos(w, delta_w)

        error_train = calcular_error_norma(Y_train, Yc_train)

        # --- Validación (sin actualizar pesos) ---
        Yc_val    = calcular_Yc(X_val, w, activacion)
        error_val = calcular_error_norma(Y_val, Yc_val)

        hist_train.append(error_train)
        hist_val.append(error_val)

        print(f"    Época {g+1:3d} | error_train={error_train:.6f}  error_val={error_val:.6f}")

        # Guardar mejor punto (menor error_val)
        if error_val < mejor_error_val:
            mejor_error_val   = error_val
            mejor_error_train = error_train
            mejores_pesos     = w.copy()

        # --- Early Stopping: error_val sube por primera vez ---
        if g > 0 and error_val > error_val_anterior:
            print(f"    *** EARLY STOP en época {g+1}: error_val subió de {error_val_anterior:.6f} a {error_val:.6f} ***")
            epoca_stop = g + 1
            break

        error_val_anterior = error_val

    return mejores_pesos, mejor_error_train, mejor_error_val, epoca_stop, hist_train, hist_val

# =====================================================================
# VALIDACIÓN CRUZADA K-FOLD
# =====================================================================

def validacion_cruzada(X, Y, k, lr_list, generaciones, activacion):
    """
    Ejecuta K rondas de validación cruzada.
    En cada ronda: K-1 folds entrenan, 1 fold valida.
    
    Retorna tabla de resultados con error_total por ronda y lr.
    """
    folds_X, folds_Y = crear_folds(X, Y, k)
    letras = "ABCDEFGHIJ"
    n_pesos = len(X[0])

    # tabla_resultados[lr] = lista de dicts por ronda
    tabla_resultados = {}

    for lr in lr_list:
        print(f"\n{'#'*60}")
        print(f"  LEARNING RATE: {lr}")
        print(f"{'#'*60}")

        resultados_rondas = []

        for i_val in range(k):
            # Fold de validación
            fold_val_letra = letras[i_val]

            # Folds de entrenamiento = todos menos el de validación
            folds_train_letras = [letras[j] for j in range(k) if j != i_val]
            descripcion_train  = "+".join(folds_train_letras)

            print(f"\n  --- Ronda {i_val+1}: Train=[{descripcion_train}]  Val=[{fold_val_letra}] ---")

            # Construir conjuntos
            X_train, Y_train = [], []
            for j in range(k):
                if j != i_val:
                    X_train.extend(folds_X[j])
                    Y_train.extend(folds_Y[j])

            X_val = folds_X[i_val]
            Y_val = folds_Y[i_val]

            n_train = len(X_train)
            n_val   = len(X_val)

            # Pesos iniciales aleatorios
            w_inicial = cargar_pesos(n_pesos)
            print(f"    Pesos iniciales: {w_inicial}")

            # Entrenar con early stopping
            mejores_pesos, e_train, e_val, epoca_stop, hist_train, hist_val = entrenar_con_early_stopping(
                X_train, Y_train, X_val, Y_val,
                w_inicial, lr, generaciones, activacion
            )

            # Error total ponderado
            error_total = (n_train * e_train + n_val * e_val) / (n_train + n_val)

            resultados_rondas.append({
                "ronda":        i_val + 1,
                "train":        descripcion_train,
                "val":          fold_val_letra,
                "n_train":      n_train,
                "n_val":        n_val,
                "error_train":  e_train,
                "error_val":    e_val,
                "error_total":  error_total,
                "epoca_stop":   epoca_stop,
                "pesos":        mejores_pesos,
                "hist_train":   hist_train,
                "hist_val":     hist_val,
            })

        tabla_resultados[lr] = resultados_rondas

    return tabla_resultados

# =====================================================================
# TABLA DE RESULTADOS
# =====================================================================

def imprimir_tabla(tabla_resultados):
    print("\n")
    print("=" * 100)
    print("TABLA DE RESULTADOS — VALIDACIÓN CRUZADA")
    print("=" * 100)
    print(f"{'LR':>6} | {'Ronda':>5} | {'Train':>5} | {'Val':>3} | {'n_train':>7} | {'n_val':>5} | {'e_train':>10} | {'e_val':>10} | {'e_total':>10} | {'Época stop':>10}")
    print("-" * 100)

    mejor_total  = float("inf")
    mejor_config = None

    for lr, rondas in tabla_resultados.items():
        for r in rondas:
            marca = ""
            if r["error_total"] < mejor_total:
                mejor_total  = r["error_total"]
                mejor_config = (lr, r)

            print(f"{lr:>6} | {r['ronda']:>5} | {r['train']:>5} | {r['val']:>3} | {r['n_train']:>7} | {r['n_val']:>5} | "
                  f"{r['error_train']:>10.6f} | {r['error_val']:>10.6f} | {r['error_total']:>10.6f} | {r['epoca_stop']:>10}")
        print("-" * 100)

    print("=" * 100)
    lr_ganador, ronda_ganadora = mejor_config
    print(f"\n  ★  MODELO GANADOR: LR={lr_ganador} | Ronda {ronda_ganadora['ronda']} "
          f"(Train={ronda_ganadora['train']}, Val={ronda_ganadora['val']}) | "
          f"error_total={ronda_ganadora['error_total']:.6f}")
    print(f"     Pesos: {ronda_ganadora['pesos']}")
    print("=" * 100)

    return lr_ganador, ronda_ganadora

# =====================================================================
# GRÁFICAS
# =====================================================================

def graficar_curvas(tabla_resultados):
    os.makedirs(PATH, exist_ok=True)

    for lr, rondas in tabla_resultados.items():
        fig, axes = plt.subplots(1, len(rondas), figsize=(6 * len(rondas), 4), sharey=False)
        if len(rondas) == 1:
            axes = [axes]

        fig.suptitle(f"Curvas de error — lr={lr}", fontsize=13)

        for ax, r in zip(axes, rondas):
            epocas = range(1, len(r["hist_train"]) + 1)
            ax.plot(epocas, r["hist_train"], label="error_train", color="steelblue")
            ax.plot(epocas, r["hist_val"],   label="error_val",   color="tomato", linestyle="--")
            ax.axvline(x=r["epoca_stop"], color="gray", linestyle=":", label=f"stop época {r['epoca_stop']}")
            ax.set_title(f"Ronda {r['ronda']}: Train={r['train']} Val={r['val']}")
            ax.set_xlabel("Época")
            ax.set_ylabel("Error L2")
            ax.legend(fontsize=8)
            ax.grid(True)

        plt.tight_layout()
        fname = os.path.join(PATH, f"curvas_lr{str(lr).replace('.','')}.png")
        plt.savefig(fname)
        print(f"  Gráfica guardada: {fname}")
        plt.close()

# =====================================================================
# MAIN
# =====================================================================

def main():
    dataset  = cargar_data()
    X, Y     = calcular_XyY(dataset)
    lr_list  = cargar_lr()

    tabla = validacion_cruzada(X, Y, K, lr_list, GENERACIONES, ACTIVACION)

    lr_ganador, ronda_ganadora = imprimir_tabla(tabla)

    graficar_curvas(tabla)

if __name__ == "__main__":
    main()


# =============================================================================
# GUÍA DE SELECCIÓN DE FUNCIÓN DE ACTIVACIÓN
# =============================================================================
#
# ¿Cómo elegir? Observa la columna Y (última columna del dataset)
#
# IDENTIDAD  →  Y continua, puede ser negativa, sin límite
# SIGMOIDE   →  Y solo tiene valores 0 o 1  (clasificación binaria)
# RELU       →  Y continua, siempre >= 0, puede tener ceros exactos
# TANH       →  Y acotada entre -1 y 1  (regresión normalizada o bipolar)
# ESCALÓN    →  Y solo 0 o 1, sin gradiente suave
# =============================================================================
