import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from generacion_poblacion import generar_poblacion_inicial
from fitness import calcular_fitness
from emparejamiento import emparejar_poblacion
from cruza import cruzar_padres
from mutacion import mutar_individuo
from poda import podar_poblacion
from graficacion import graficar_sube_baja


def crear_video_desde_frames(frames_dir, output_path, fps=15):
    
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if not frames:
        print("No se encontraron frames para crear el video")
        return
    
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    video.release()
    print(f"\nVideo creado exitosamente: {output_path}")


def algoritmo_genetico_sube_baja(generaciones=50, tasa_mutacion=0.1):
    
    distancias = [-2.5, -0.833, 0.833, 2.5]
    
    pesos = [0.0]  # Índice 0 = asiento vacío
    pesos.extend(np.random.uniform(60, 85, 10).tolist())  # Índices 1-10
    
    print("Pesos generados:")
    for i, peso in enumerate(pesos):
        if i == 0:
            print(f"  Persona {i}: {peso:.2f} kg (VACÍO)")
        else:
            print(f"  Persona {i}: {peso:.2f} kg")
    print()
    
    # Crear directorio para frames
    frames_dir = 'frames'
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    poblacion = generar_poblacion_inicial()
    
    historial_fitness = []
    historial_mejor_individuo = []
    
    for gen in range(generaciones):
    
        fitness_values = [calcular_fitness(ind, pesos, distancias) for ind in poblacion]
        
        mejor_idx = min(range(len(fitness_values)), key=lambda i: abs(fitness_values[i]))
        mejor_individuo = poblacion[mejor_idx]
        mejor_fitness = fitness_values[mejor_idx]
        
        historial_fitness.append(mejor_fitness)
        historial_mejor_individuo.append(mejor_individuo.copy())
        
        if gen % 1 == 0:
            frame_path = f'{frames_dir}/frame_{gen:04d}.png'
            graficar_sube_baja(mejor_individuo, pesos, distancias, gen, mejor_fitness, frame_path)
        
        if gen % 50 == 0 or gen == generaciones - 1:
            print(f"Generación {gen}: Mejor torque = {mejor_fitness:.4f} N·m, Individuo = {mejor_individuo}")
        
        # Emparejamiento
        parejas = emparejar_poblacion(poblacion, fitness_values)
        
        # Cruza
        nueva_poblacion = []
        for padre1, padre2 in parejas:
            hijo1, hijo2 = cruzar_padres(padre1, padre2)
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Mutación
        nueva_poblacion = [mutar_individuo(ind, pesos, tasa_mutacion) 
                          for ind in nueva_poblacion]
        
        # Combinar población antigua y nueva
        poblacion_combinada = poblacion + nueva_poblacion
        fitness_combinados = fitness_values + [calcular_fitness(ind, pesos, distancias) 
                                               for ind in nueva_poblacion]
        
        # Poda: mantener los 35 mejores
        poblacion, fitness_values = podar_poblacion(poblacion_combinada, 
                                                     fitness_combinados, 
                                                     tam_objetivo=35)
    
    # Resultado final
    mejor_idx = min(range(len(fitness_values)), key=lambda i: abs(fitness_values[i]))
    mejor_individuo_final = poblacion[mejor_idx]
    mejor_fitness_final = fitness_values[mejor_idx]
    
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    print(f"Mejor individuo: {mejor_individuo_final}")
    print(f"Torque final: {mejor_fitness_final:.4f} N·m")
    print(f"Pesos en asientos: {[pesos[i] for i in mejor_individuo_final]}")
    print("="*60)
    
    return mejor_individuo_final, mejor_fitness_final, historial_fitness, pesos


# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    print("Iniciando Algoritmo Genético para el Sube y Baja")
    print("="*60)
    
    # Ejecutar algoritmo genético
    mejor_ind, mejor_fit, historial, pesos = algoritmo_genetico_sube_baja(
        generaciones=50, 
        tasa_mutacion=0.1
    )
    
    # Crear video
    crear_video_desde_frames(
        frames_dir='frames',
        output_path='evolucion_sube_baja.mp4',
        fps=15
    )
    
    # Crear gráfica de evolución del fitness
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(historial)), [abs(f) for f in historial], 'b-', linewidth=2)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('|Torque| (N·m)', fontsize=12)
    plt.title('Evolución del Mejor Fitness a lo largo de las Generaciones', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evolucion_fitness.png', dpi=150)
    plt.close()
    
    print("\nArchivos generados:")
    print("- evolucion_sube_baja.mp4: Video con la evolución")
    print("- evolucion_fitness.png: Gráfica de evolución del fitness")