import random

def generar_poblacion_inicial():
    poblacion = []
    tam_poblacion = 35
    num_asientos = 4
    num_personas = 11  
    
    for _ in range(tam_poblacion):
        individuo = random.sample(range(num_personas), num_asientos)
        poblacion.append(individuo)
    
    return poblacion