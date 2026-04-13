import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
#  PASO 1 — SELECCIONA TU DATASET
#  Descomenta UNO y comenta los otros dos
# ─────────────────────────────────────────────

data = np.loadtxt("C233435.csv", delimiter=",")

PROBLEMA = "regresion"   # <-- indica qué tipo de problema es



# ─────────────────────────────────────────────
#  PASO 2 — SEPARAR FEATURES Y TARGET
#  (esto nunca cambia)
# ─────────────────────────────────────────────

X = data[:, :-1]   # todas las columnas menos la última
y = data[:, -1]    # solo la última columna

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

n_features = X_train.shape[1]

# ─────────────────────────────────────────────
#  PASO 3 — ARQUITECTURA SEGÚN EL PROBLEMA
# ─────────────────────────────────────────────

if PROBLEMA == "binario":
    output_layer    = keras.layers.Dense(1, activation='sigmoid')
    loss_fn         = 'binary_crossentropy'
    metric          = 'accuracy'

elif PROBLEMA == "regresion":
    output_layer    = keras.layers.Dense(1, activation='linear')
    loss_fn         = 'mse'
    metric          = 'mae'

model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8,  activation='relu'),
    output_layer,
])

model.summary()

# ─────────────────────────────────────────────
#  PASO 4 — COMPILAR Y ENTRENAR
# ─────────────────────────────────────────────

model.compile(optimizer='adam', loss=loss_fn, metrics=[metric])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.1,
    verbose=1
)

# ─────────────────────────────────────────────
#  PASO 5 — EVALUAR
# ─────────────────────────────────────────────


resultados = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test).flatten()

if PROBLEMA == "binario":
    print(f"\nLoss:     {resultados[0]:.4f}")
    print(f"Accuracy: {resultados[1]:.2%}")
elif PROBLEMA == "regresion":
    print(f"\nLoss (MSE): {resultados[0]:.2f}")
    print(f"MAE:        {resultados[1]:.2f}")

# ─────────────────────────────────────────────
#  PASO 6 — GRÁFICAS DE REPORTE
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Reporte de entrenamiento", fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── 1. Evolución del loss ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'],     label='Train',      linewidth=1.5)
ax1.plot(history.history['val_loss'], label='Validación', linewidth=1.5, linestyle='--')
ax1.set_title('Evolución del error (loss)')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss (MSE)' if PROBLEMA == "regresion" else 'Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── 2. Evolución de la métrica (MAE o Accuracy) ───────────
ax2 = fig.add_subplot(gs[0, 1])
metric_key     = 'mae'       if PROBLEMA == "regresion" else 'accuracy'
val_metric_key = 'val_mae'   if PROBLEMA == "regresion" else 'val_accuracy'
metric_label   = 'MAE'       if PROBLEMA == "regresion" else 'Accuracy'

ax2.plot(history.history[metric_key],     label='Train',      linewidth=1.5)
ax2.plot(history.history[val_metric_key], label='Validación', linewidth=1.5, linestyle='--')
ax2.set_title(f'Evolución del {metric_label}')
ax2.set_xlabel('Época')
ax2.set_ylabel(metric_label)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── 3. Predicciones vs valores reales (solo regresión) ────
ax3 = fig.add_subplot(gs[0, 2])
if PROBLEMA == "regresion":
    ax3.scatter(y_test, y_pred, alpha=0.6, s=20, label='Muestras')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax3.plot(lims, lims, 'r--', linewidth=1, label='Predicción ideal')
    ax3.set_title('Predicciones vs valores reales')
    ax3.set_xlabel('Valor real')
    ax3.set_ylabel('Predicción')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred_bin = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_bin)
    ConfusionMatrixDisplay(cm).plot(ax=ax3, colorbar=False)
    ax3.set_title('Matriz de confusión')

# ── 4. Distribución de residuos (solo regresión) ──────────
ax4 = fig.add_subplot(gs[1, 0])
if PROBLEMA == "regresion":
    residuos = y_pred - y_test
    ax4.hist(residuos, bins=30, edgecolor='white', linewidth=0.5)
    ax4.axvline(0, color='red', linestyle='--', linewidth=1.2)
    ax4.set_title('Distribución de residuos')
    ax4.set_xlabel('Error (predicción − real)')
    ax4.set_ylabel('Frecuencia')
    ax4.grid(True, alpha=0.3)
else:
    ax4.axis('off')  # No aplica para clasificación

# ── 5. Histograma de pesos por capa ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        w = weights[0].flatten()
        ax5.hist(w, bins=40, alpha=0.6, label=layer.name, edgecolor='none')
ax5.set_title('Distribución de pesos por capa')
ax5.set_xlabel('Valor del peso')
ax5.set_ylabel('Frecuencia')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── 6. Gap train vs validación (overfitting) ──────────────
ax6 = fig.add_subplot(gs[1, 2])
train_loss = np.array(history.history['loss'])
val_loss   = np.array(history.history['val_loss'])
gap        = val_loss - train_loss
epochs_range = np.arange(1, len(gap) + 1)

ax6.plot(epochs_range, gap, color='orange', linewidth=1.5)
ax6.axhline(0, color='gray', linestyle='--', linewidth=1)
ax6.fill_between(epochs_range, 0, gap,
                 where=(gap > 0), alpha=0.2, color='red',   label='Overfitting')
ax6.fill_between(epochs_range, 0, gap,
                 where=(gap < 0), alpha=0.2, color='green', label='Underfitting')
ax6.set_title('Gap de generalización')
ax6.set_xlabel('Época')
ax6.set_ylabel('val_loss − train_loss')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.savefig("reporte_entrenamiento.png", dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada como reporte_entrenamiento.png")