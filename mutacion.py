import random

def mutar_individuo(individuo, pesos, tasa_mutacion=0.1):
    
    individuo_mutado = individuo.copy()
    
    # Decidir si ocurre mutación
    if random.random() < tasa_mutacion:
        # Seleccionar posición aleatoria a mutar
        posicion = random.randint(0, 3)
        persona_actual = individuo_mutado[posicion]
        peso_actual = pesos[persona_actual]
        
        # Encontrar personas disponibles (no están en el individuo)
        disponibles = [i for i in range(11) if i not in individuo_mutado]
        
        if disponibles:
            # Encontrar la persona más lejana en peso
            diferencias = [(i, abs(pesos[i] - peso_actual)) for i in disponibles]
            persona_lejana = max(diferencias, key=lambda x: x[1])[0]
            
            # Realizar la mutación
            individuo_mutado[posicion] = persona_lejana
    
    return individuo_mutado