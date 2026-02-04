import random
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import copy
from individuo import Individuo
from generacion import GENERACIONES_TOTALES, GENERACION_ACTUAL, ID, UMBRAL_MUTA, PESOS_DISPONIBLES, POBLACION_MAX


def cargar_personas():
    
    pesos = PESOS_DISPONIBLES

    personas = []

    for i, peso in enumerate(pesos):
        ind = Individuo(
            id=None,
            genes=[{"peso": peso, "distancia": None}],
            generacion=GENERACION_ACTUAL,
            torque=None,
            prob_muta=None
        )
        personas.append(ind)


    return personas

def cargar_distancia_asientos():
    distancias = [-2.5, -0.83, 0.83, 2.5]
    return distancias

def crear_individuos(personas, distancias):
    global ID
    individuos = []

    personas_disp = personas.copy()
    genes = []

    for d in distancias:
        per = random.choice(personas_disp)
        peso = per.genes[0]["peso"]

        gen = {
            "peso": peso,
            "distancia": d
        }

        genes.append(gen)
        personas_disp.remove(per)

        individuo = Individuo(
            id=ID,
            genes=genes,
            generacion=GENERACION_ACTUAL,
            torque=None,
            prob_muta=None
        )

        individuos.append(individuo)

        ID += 1
    
    return individuos
   
def calcular_fitness(individuos):
    for individuo in individuos:
        torque_total = 0
        genes_actualizados = []

        for gen in individuo.genes:
            torque = gen["peso"] * gen["distancia"]
            torque_total += torque

            gen_actualizado = {
                "peso": gen["peso"],
                "distancia": gen["distancia"],
                "torque": torque
            }
            genes_actualizados.append(gen_actualizado)

        individuo.genes = genes_actualizados
        individuo.torque = torque_total

    return individuos
              
def emparejar(individuos):
    individuos_ordenados = sorted(
        individuos,
        key=lambda ind: abs(ind.torque)
    )

    n = len(individuos_ordenados)

    if n % 2 != 0:
        individuos_ordenados.pop(n // 2)
        n -= 1

    mitad = n // 2

    seg_A = individuos_ordenados[:mitad]   
    seg_B = individuos_ordenados[mitad:]    

    parejas = []

    for a, b in zip(seg_A, seg_B):
        parejas.append((a, b))

    return parejas
    
def cruzar(parejas):
    global ID

    nueva_poblacion = []

    for padre1, padre2 in parejas:
        nueva_poblacion.append(padre1)
        nueva_poblacion.append(padre2)

        genes_p1 = padre1.genes
        genes_p2 = padre2.genes

        punto_corte = len(genes_p1) // 2

        genes_hijo1 = copy.deepcopy(
            genes_p1[:punto_corte] + genes_p2[punto_corte:]
        )

        genes_hijo2 = copy.deepcopy(
            genes_p2[:punto_corte] + genes_p1[punto_corte:]
        )

        hijo1 = Individuo(
            id=ID,
            genes=genes_hijo1,
            generacion=GENERACION_ACTUAL + 1,
            torque=None,
            prob_muta=None
        )
        ID += 1

        hijo2 = Individuo(
            id=ID,
            genes=genes_hijo2,
            generacion=GENERACION_ACTUAL + 1,
            torque=None,
            prob_muta=None
        )
        ID += 1

        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)
    
    return nueva_poblacion

def mutar(poblacion):
    
    for individuo in poblacion:
        individuo.prob_muta = random.uniform(0, 1)

        if individuo.prob_muta <= UMBRAL_MUTA:
            genes = individuo.genes

            gen_a_mutar = random.choice(genes)

            pesos_usados = {g["peso"] for g in genes}

            pesos_libres = [
                p for p in PESOS_DISPONIBLES if p not in pesos_usados
            ]

            if pesos_libres:
                nuevo_peso = random.choice(pesos_libres)
                gen_a_mutar["peso"] = nuevo_peso

    return poblacion

def podar(poblacion):
    random.shuffle(poblacion)
    
    return poblacion[:POBLACION_MAX]

def seleccionar_mejor_individuo(poblacion):
    poblacion_ordenada = sorted(
        poblacion,
        key=lambda ind: abs(ind.torque)
    )
    """ print(f"mejor generacional (gen: {GENERACION_ACTUAL}): {poblacion_ordenada[0]}")  """
    return poblacion_ordenada[0]

def graficar_mejor_fitness(historial_fitness, generacion):
    plt.figure()
    plt.plot(historial_fitness, marker="o")
    plt.xlabel("Generación")
    plt.ylabel("|Torque| (Fitness)")
    plt.title("Convergencia al equilibrio (Torque → 0)")
    plt.grid(True)

    nombre_archivo = f"img/frame_{generacion:03d}.png"
    plt.savefig(nombre_archivo)
    plt.close()
    
def crear_video():
    ruta = "img"
    imagenes = sorted(
        [img for img in os.listdir(ruta) if img.endswith(".png")]
    )

    frame = cv2.imread(os.path.join(ruta, imagenes[0]))
    alto, ancho, _ = frame.shape

    video = cv2.VideoWriter(
        "img/convergencia_algoritmo_genetico.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,  # fps
        (ancho, alto)
    )

    for img in imagenes:
        frame = cv2.imread(os.path.join(ruta, img))
        video.write(frame)

    video.release()

def main():  
    os.makedirs("img", exist_ok=True)
    global GENERACION_ACTUAL
    historial_mejor_fitness = []
    
    personas=cargar_personas()
    distancias_asientos=cargar_distancia_asientos()
        
    poblacion= crear_individuos(personas, distancias_asientos)
    
    print("\n" + "="*80)
    print("MEJOR INDIVIDUO POR GENERACIÓN")
    print("="*80)
    print(f"{'Gen del ind.':<12} {'ID':<5} {'Torque':<12} {'Distribución de Pesos (kg)'}")
    print("-"*80)
    
    for _ in range(GENERACIONES_TOTALES):
    
        poblacion = calcular_fitness(poblacion)
    
        parejas = emparejar(poblacion)
        
        hijos_y_padres = cruzar(parejas)

        nueva_poblacion_c_fitness= calcular_fitness(hijos_y_padres) 
    
        mutados = mutar(nueva_poblacion_c_fitness)
    
        if len(mutados) >= POBLACION_MAX:
            poblacion = podar(mutados)
        else:
            poblacion = mutados
            
        mejor = seleccionar_mejor_individuo(poblacion)
        
        pesos = [gen["peso"] for gen in mejor.genes]
        
        print(f"{mejor.generacion:<12} {mejor.id:<5} {mejor.torque:<12.2f} {pesos}")


        fitness_actual = abs(mejor.torque)
        historial_mejor_fitness.append(fitness_actual)

        graficar_mejor_fitness(historial_mejor_fitness, GENERACION_ACTUAL)
        GENERACION_ACTUAL+=1
    print()
    crear_video()


    
    
if __name__ == "__main__":
    main()