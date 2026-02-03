def cruzar_padres(padre1, padre2):
    """
    Cruza dos padres para generar dos hijos.
    Alterna genes entre padres (p1, p2, p1, p2) y corrige duplicados.
    
    Parámetros:
    - padre1: Lista de 4 índices
    - padre2: Lista de 4 índices
    
    Retorna:
    - hijo1, hijo2: Tupla con los dos hijos generados
    """
    # Crear hijos alternando genes
    hijo1 = [padre1[0], padre2[1], padre1[2], padre2[3]]
    hijo2 = [padre2[0], padre1[1], padre2[2], padre1[3]]
    
    # Función auxiliar para corregir duplicados
    def corregir_duplicados(hijo):
        usado = set()
        for i in range(len(hijo)):
            if hijo[i] in usado:
                # Buscar un valor no usado (de 0 a 10)
                for nuevo_valor in range(11):
                    if nuevo_valor not in usado:
                        hijo[i] = nuevo_valor
                        break
            usado.add(hijo[i])
        return hijo
    
    hijo1 = corregir_duplicados(hijo1)
    hijo2 = corregir_duplicados(hijo2)
    
    return hijo1, hijo2