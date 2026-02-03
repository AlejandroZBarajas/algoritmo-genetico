def podar_poblacion(poblacion, fitness_values, tam_objetivo=35):
    
    indices_ordenados = sorted(range(len(fitness_values)), 
                               key=lambda i: abs(fitness_values[i]))
    
    # Seleccionar los mejores
    indices_mejores = indices_ordenados[:tam_objetivo]
    
    poblacion_podada = [poblacion[i] for i in indices_mejores]
    fitness_podados = [fitness_values[i] for i in indices_mejores]
    
    return poblacion_podada, fitness_podados