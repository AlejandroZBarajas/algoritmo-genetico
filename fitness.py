def calcular_fitness(individuo, pesos, distancias):
    torque = 0
    
    for i in range(4):
        persona_idx = individuo[i]
        peso = pesos[persona_idx]
        distancia = distancias[i]
        torque += peso * distancia
    
    return torque