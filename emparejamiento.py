def emparejar_poblacion(poblacion, fitness_values):

    indices_ordenados = sorted(range(len(fitness_values)), 
                               key=lambda i: abs(fitness_values[i]))
    poblacion_ordenada = [poblacion[i] for i in indices_ordenados]
    
    # Dividir en dos segmentos
    tam_poblacion = len(poblacion_ordenada)
    punto_medio = tam_poblacion // 2
    
    segmento_a = poblacion_ordenada[:punto_medio]
    segmento_b = poblacion_ordenada[punto_medio:]
    
    # Emparejar primer elemento de A con primer elemento de B, etc.
    parejas = []
    for i in range(len(segmento_a)):
        if i < len(segmento_b):  # Asegurar que hay pareja en segmento B
            parejas.append((segmento_a[i], segmento_b[i]))
    
    return parejas