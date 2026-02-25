import random
import math
import matplotlib.pyplot as plt
import os

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

    #dataset, vocabulario = codificar_strings(dataset)
    #print("Diccionario:", vocabulario)
    
    return dataset
    
def codificar_strings(dataset):
    diccionario = {}
    contador = 0

    dataset_numerico = []

    for fila in dataset:
        fila_num = []
        for valor in fila:
            if isinstance(valor, str):
                if valor not in diccionario:
                    diccionario[valor] = contador
                    contador += 1
                fila_num.append(diccionario[valor])
            else:
                fila_num.append(valor)
        dataset_numerico.append(fila_num)

    return dataset_numerico, diccionario
  
def calcular_XyY(dataset):
    # 1. separar etiquetas
    y = [row[-1] for row in dataset]

    # 2. extraer solo features (sin y)
    X_sin_bias = [row[:-1] for row in dataset]

    # 3. normalizar features
    X_norm = normalizar_minmax(X_sin_bias)

    # 4. agregar bias
    X_final = [[1] + row for row in X_norm]

    print(f"Filas de X (muestras): {len(X_final)}")
    print(f"Columnas de X (features + bias): {len(X_final[0])}")

    return X_final, y
    
def normalizar_minmax(X):
    X_norm = []
    num_features = len(X[0])

    mins = [min(row[j] for row in X) for j in range(num_features)]
    maxs = [max(row[j] for row in X) for j in range(num_features)]

    for row in X:
        norm_row = []
        for j in range(num_features):
            if maxs[j] == mins[j]:
                norm_row.append(0.0)
            else:
                norm_row.append((row[j] - mins[j]) / (maxs[j] - mins[j]))
        X_norm.append(norm_row)

    return X_norm

def cargar_pesos(n):
    #se cargan n pesos aleatorios entre -1 y 1 usando 2 decimales
    pesos = [round(random.uniform(-1, 1), 2) for _ in range(n)]
    print(f"pesos cargados:  {pesos}")
    return pesos

def calcular_Yc(X,W, activacion):   
    Yc=[]
    for fila in X:
        u=0
        for j in range(len(W)):            
            u += fila[j] * W[j]
            
        y= activar_funcion(u, activacion)
        Yc.append(u)
    return Yc
        
""" def calcular_E(Y, Yc):
    e = []
    for y, yc in zip(Y, Yc):
        e.append(y - yc)
    return e """
def calcular_E(Y, Yc):
    return [yc - y for y, yc in zip(Y, Yc)]

def calcular_delta_w(matriz_x,errores,lr):
    n_pesos = len(matriz_x[0])  
    delta_w = []

    for j in range(n_pesos):
        suma = 0
        for i in range(len(errores)):
            suma += matriz_x[i][j] * errores[i]
        delta_w.append(lr * suma / len(errores))

    return delta_w

def actualizar_pesos(w, delta_w):
    for i in range(len(w)):
        w[i] -= delta_w[i]
    return w


def cargar_lr():
    lr = [0.01, 0.02, 0.03, 0.04, 0.05]
    return lr

def activar_funcion(u, activacion):
    
    # ADALINE / regresión lineal
    if activacion == "identidad":
        return u

    # Perceptrón clásico (escalón)
    elif activacion == "escalon":
        if u < 0:
            return 0
        else:
            return 1


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

def graficar_error(errores_por_lr):
    path="./algoritmo-genetico/neurona/"
    plt.figure(figsize=(8,5))

    for lr, errores in errores_por_lr.items():
        plt.plot(range(1, len(errores)+1), errores, label=f"lr={lr}")

    plt.xlabel("Generaciones")
    plt.ylabel("Norma L2 del error ||E||")

    plt.title("Evolución del error para distintos learning rates")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path,"error.png"))

###########################################################################
###########################################################################
###########################################################################


def graficar_pesos(pesos_por_lr, indice_peso=0):
    path="./algoritmo-genetico/neurona/"
    plt.figure(figsize=(8,5))

    for lr, pesos_hist in pesos_por_lr.items():
        peso_i = [w[indice_peso] for w in pesos_hist]
        plt.plot(range(1, len(peso_i)+1), peso_i, label=f"lr={lr}")

    plt.xlabel("Generaciones")
    plt.ylabel(f"Peso w{indice_peso}")
    plt.title(f"Evolución del peso w{indice_peso}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path,"peso.png"))




###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################


def main():
    generaciones = 30
    
    dataset=cargar_data()

    matriz_x, matriz_y= calcular_XyY(dataset)    

    n_pesos = len(matriz_x[0])
    
    l_rates=cargar_lr()
    
    errores_por_lr = {}
    pesos_por_lr = {}
    
    for lr in l_rates:
        print("\n==============================")
        print("Learning rate:", lr)
        w=cargar_pesos(n_pesos)
 
        
        errores_hist = []
        pesos_hist = []
        
        for g in range(generaciones):
            Yc = calcular_Yc(matriz_x,w, "identidad")   ##    escalon o identidad
            errores=calcular_E(matriz_y, Yc)
            delta_w=calcular_delta_w(matriz_x, errores,lr)
            w = actualizar_pesos(w, delta_w)
            
            error_norm = math.sqrt(sum(e**2 for e in errores))
            ####         HISTORIALES
            errores_hist.append(error_norm)
            pesos_hist.append(w.copy())  
            
            print(f"Gen {g+1} | Error total: {error_norm:.4f}")

        errores_por_lr[lr] = errores_hist
        pesos_por_lr[lr] = pesos_hist
    
    graficar_error(errores_por_lr)
    graficar_pesos(pesos_por_lr)

    
    
    
if __name__ == "__main__":
    main()

 
 