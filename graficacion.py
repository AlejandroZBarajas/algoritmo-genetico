import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def graficar_sube_baja(individuo, pesos, distancias, generacion, torque, frame_path):
    """
    Genera una visualización del sube y baja con el mejor individuo.
    
    Parámetros:
    - individuo: Mejor individuo de la generación
    - pesos: Lista de pesos
    - distancias: Lista de distancias de los asientos
    - generacion: Número de generación actual
    - torque: Valor del torque del individuo
    - frame_path: Ruta donde guardar el frame
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Configurar límites y aspecto
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Dibujar el fulcro (triángulo)
    fulcro = patches.Polygon([[0, -0.5], [-0.3, 0], [0.3, 0]], 
                            closed=True, color='gray', zorder=5)
    ax.add_patch(fulcro)
    
    # Calcular el ángulo de inclinación basado en el torque
    # Ángulo proporcional al torque (limitado para visualización)
    max_torque = 500  # Valor para normalizar
    angulo = np.clip(torque / max_torque * 15, -15, 15)  # Máximo 15 grados
    angulo_rad = np.radians(angulo)
    
    # Dibujar la barra del sube y baja (rotada)
    barra_length = 5
    x_barra = [-barra_length/2, barra_length/2]
    y_barra = [0, 0]
    
    # Rotar la barra
    x_rot = [x * np.cos(angulo_rad) - y * np.sin(angulo_rad) for x, y in zip(x_barra, y_barra)]
    y_rot = [x * np.sin(angulo_rad) + y * np.cos(angulo_rad) for x, y in zip(x_barra, y_barra)]
    
    ax.plot(x_rot, y_rot, 'brown', linewidth=8, zorder=3, label='Barra')
    
    # Dibujar los asientos y personas
    colores = ['red', 'blue', 'green', 'orange']
    for i, dist in enumerate(distancias):
        persona_idx = individuo[i]
        peso = pesos[persona_idx]
        
        # Calcular posición del asiento (rotada)
        x_asiento = dist * np.cos(angulo_rad)
        y_asiento = dist * np.sin(angulo_rad)
        
        # Dibujar asiento
        asiento = patches.Circle((x_asiento, y_asiento), 0.15, 
                                color='black', zorder=4)
        ax.add_patch(asiento)
        
        # Dibujar persona si no está vacío
        if persona_idx != 0:
            persona = patches.Circle((x_asiento, y_asiento + 0.4), 0.25, 
                                    color=colores[i], alpha=0.7, zorder=6)
            ax.add_patch(persona)
            
            # Etiqueta con peso
            ax.text(x_asiento, y_asiento + 0.8, f'{peso:.1f}kg', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            # Asiento vacío
            ax.text(x_asiento, y_asiento + 0.4, 'VACÍO', 
                   ha='center', va='center', fontsize=9, style='italic')
    
    # Título y información
    ax.set_title(f'Generación {generacion} - Mejor Individuo\n' + 
                f'Torque: {torque:.2f} N·m', 
                fontsize=16, fontweight='bold')
    
    # Información adicional
    info_text = f'Individuo: {individuo}\n'
    info_text += f'Pesos: {[f"{pesos[i]:.1f}" if i != 0 else "0.0" for i in individuo]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Indicador de equilibrio
    if abs(torque) < 1:
        equilibrio_color = 'green'
        equilibrio_text = '¡EQUILIBRADO!'
    elif abs(torque) < 10:
        equilibrio_color = 'yellow'
        equilibrio_text = 'Casi equilibrado'
    else:
        equilibrio_color = 'red'
        equilibrio_text = 'Desequilibrado'
    
    ax.text(0.98, 0.02, equilibrio_text, transform=ax.transAxes, 
           fontsize=14, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor=equilibrio_color, alpha=0.7),
           fontweight='bold')
    
    ax.set_xlabel('Distancia desde el fulcro (m)', fontsize=12)
    ax.set_ylabel('Altura (m)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close()