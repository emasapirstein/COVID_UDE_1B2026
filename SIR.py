import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torchdiffeq import odeint
import os
import matplotlib.ticker as ticker

#=====================
# Semillas
#=====================
def set_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(123)
#=====================
#Configuración general
#=====================
DEVICE = "cpu"
carpeta = carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

# Crear carpeta para gráficos si no existe
os.makedirs(carpeta + "SIR", exist_ok=True)

#==================================
#Carga y Preprocesamiento de datos
#==================================
df = pd.read_csv(CSV_PATH, usecols=["fecha_apertura", "fecha_fallecimiento", "clasificacion_resumen"])

#Filtramos solo confirmados para evitar ruido
df = df[df["clasificacion_resumen"] == "Confirmado"]

DATE_COLUMN = "fecha_apertura"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

#Contamos casos diarios
cases_per_day = df.groupby(DATE_COLUMN).size().sort_index()

#Suavizamos para eliminar el efecto del fin de semana
cases_per_day = cases_per_day.rolling(7, center=True).mean().bfill().ffill()

nuevos_casos_diarios = cases_per_day.values.astype(np.float32)

#Población total aproximada de Argentina
N = 46000000

#Cantidad de días que vamos a usar para entrenar
MAX_DAYS_TRAIN = 300

#Cantidad total de días disponibles
MAX_DAYS_TOT=len(nuevos_casos_diarios)

#Casos de train
train_cases=nuevos_casos_diarios[:MAX_DAYS_TRAIN]
test_cases=nuevos_casos_diarios[MAX_DAYS_TRAIN:MAX_DAYS_TOT]

#=====================================
# Reconstruccion sintetica de I y R
#=====================================

#Asumimos un tiempo medio de recuperación de 14 días (parámetro clínico)
tau_random = int(np.random.normal(loc=14, scale=2))
tau_random = max(1, tau_random)  #evitamos valores negativos o cero

# Inicializamos vectores para las trayectorias
I_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)
R_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)

I_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)
R_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)

# Reconstrucción día por día de la cantidad total de infectados activos y recuperados
for t in range(MAX_DAYS_TRAIN):
    #Los infectados activos hoy son la suma de los que se contagiaron en los últimos TAU días
    inicio_ventana = max(0, t - tau_random + 1)
    I_sintetico_train[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])
    I_sintetico_full[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])

    # Los recuperados hoy son todos los que se contagiaron antes de esa ventana
    if t >= tau_random:
        R_sintetico_train[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])
        R_sintetico_full[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])

for t in range(MAX_DAYS_TRAIN, MAX_DAYS_TOT):
    # Para los días de test, seguimos la misma lógica pero con los datos completos
    inicio_ventana = max(0, t - tau_random + 1)
    I_sintetico_full[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])
    if t >= tau_random:
        R_sintetico_full[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])

# Normalizamos las curvas sintéticas por la población total N
I_obs_norm_train = I_sintetico_train / N
R_obs_norm_train = R_sintetico_train / N
S_obs_norm_train = 1.0 - I_obs_norm_train - R_obs_norm_train

I_obs_norm_full = I_sintetico_full / N
R_obs_norm_full = R_sintetico_full / N
S_obs_norm_full = 1.0 - I_obs_norm_full - R_obs_norm_full


# Vector de Tiempo y conversión a Tensores
t_obs = np.arange(MAX_DAYS_TRAIN, dtype=np.float32)
t_obs_torch = torch.tensor(t_obs, device=DEVICE)

I_obs_torch = torch.tensor(I_obs_norm_train, device=DEVICE)
R_obs_torch = torch.tensor(R_obs_norm_train, device=DEVICE)

#=====================
# Modelo SIR clásico
#=====================
class SIRModel(nn.Module):
    def __init__(self, beta=0.3, tau=14):
        super().__init__()

        self.beta_raw = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.tau_raw = nn.Parameter(torch.tensor(tau, dtype=torch.float32))

    def forward(self, t, u):
        S, I, R = u

        # Garantizamos positividad estricta usando softplus
        beta = torch.nn.functional.softplus(self.beta_raw)
        tau = torch.nn.functional.softplus(self.tau_raw)
        gamma=1/tau

        infection = beta * S * I
        recovery = gamma * I

        dS = -infection
        dI = infection - recovery
        dR = recovery

        return torch.stack([dS, dI, dR])

#=======================
# Condiciones iniciales
#=======================
# Inicializamos basándonos en los datos sintéticos de los primeros días, asegurando que I0 sea positivo
I0 = np.mean(I_obs_norm_train[:14])
R0 = 0.0
S0 = 1.0 - I0 - R0

y0 = torch.tensor([S0, I0, R0], dtype=torch.float32, device=DEVICE)

model = SIRModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

#=============================================================
# Entrenamiento (trajectory matching de múltiples variables)
#=============================================================
n_epochs = 500
loss_history = []
beta_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Resolvemos las ODEs para SIR
    pred_y = odeint(model, y0, t_obs_torch, method='rk4')

    # Extraemos las curvas predichas
    I_pred = pred_y[:, 1]
    R_pred = pred_y[:, 2]

    # Función de costo sobre I(t)
    loss = torch.mean((I_pred - I_obs_torch) ** 2)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    current_beta = torch.nn.functional.softplus(model.beta_raw).item()
    beta_history.append(current_beta)

    if epoch % 100 == 0:
        beta = torch.nn.functional.softplus(model.beta_raw).item()
        tau = torch.nn.functional.softplus(model.tau_raw).item()
        gamma = 1.0 / tau
        R0_efectivo = beta / gamma
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.8f} | beta: {beta:.4f} | tau: {tau:.4f} | gamma: {gamma:.4f} | R0: {R0_efectivo:.2f}")

#============================
# Solución final e Inferencia
#============================
MAX_DAYS_SIM = MAX_DAYS_TOT
t_sim = torch.arange(MAX_DAYS_SIM, dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    pred_y_long = odeint(model, y0, t_sim, method='rk4')

S_long, I_long, R_long = pred_y_long[:,0], pred_y_long[:,1], pred_y_long[:,2]

loss_test = torch.mean((I_long - I_obs_norm_full) ** 2)

# Cálculo de Errores Finales
train_mse = torch.mean((I_long[:MAX_DAYS_TRAIN] - torch.tensor(I_obs_norm_full[:MAX_DAYS_TRAIN], device=DEVICE)) ** 2).item()
test_mse = torch.mean((I_long[MAX_DAYS_TRAIN:] - torch.tensor(I_obs_norm_full[MAX_DAYS_TRAIN:], device=DEVICE)) ** 2).item()

print("\n=== Resultados del Ajuste ===")
print(f"Error Train (MSE): {train_mse:.8e}")
print(f"Error Test (MSE):  {test_mse:.8e}")

#============================
# Visualización de Resultados
#============================
# Función para formatear el eje Y en millones
def configurar_eje_y_millones(ax):
    ax.set_ylabel("Millones de Personas", fontsize=14)
    # Formatea los números como decimales simples (ej: 1.5, 2.0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}'))

# ==============================================
# Gráfico 1: Infectados (I) y Recuperados (R)
# ===============================================
fig1, ax1 = plt.subplots(figsize=(16, 8), dpi=300)

ax1.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3, label='Zona de Entrenamiento')
ax1.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_TOT, color='lightyellow', alpha=0.3, label='Zona de Test')
ax1.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle="--", linewidth=2)

# Graficamos dividiendo por 1e6 para mostrar en millones
ax1.plot(I_obs_norm_full * N, label="I Observado (Sintético)", color="#FF8C00", lw=2, alpha=0.8)
ax1.plot(I_long.cpu().numpy() * N, label="I Predicho (Modelo SIR)", color="#8B0000", linestyle="--", lw=3)
ax1.plot(R_obs_norm_full * N, label="R Observado (Sintético)", color="#32CD32", lw=2, alpha=0.8)
ax1.plot(R_long.cpu().numpy() * N, label="R Predicho (Modelo SIR)", color="#006400", linestyle="--", lw=3)

ax1.set_xlabel("Días Transcurridos", fontsize=14)
configurar_eje_y_millones(ax1)
ax1.set_title("Infectados y Recuperados (Escala en Millones)", fontsize=18, fontweight='bold')
ax1.legend(fontsize=12, loc="upper left")
ax1.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
ruta_grafico1 = carpeta + "graficosSIR/infectados_recuperados_millones.png"
fig1.savefig(ruta_grafico1)

# ================================
# Gráfico 2: Evolución de la Loss 
# ================================
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

ax2.plot(loss_history, color="purple", lw=2)
ax2.set_yscale('log')
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("MSE Loss (escala log)", fontsize=12)
ax2.set_title("Convergencia del Error (Loss)", fontsize=14, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
ruta_grafico2 = carpeta + "graficosSIR/evolucion_loss.png"
fig2.savefig(ruta_grafico2)

# =============================
# Gráfico 3: Evolución de Beta 
# =============================
fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)

ax3.plot(beta_history, color="teal", lw=2)
ax3.set_xlabel("Epochs", fontsize=12)
ax3.set_ylabel(r"Valor de $\beta$", fontsize=12)
ax3.set_title(r"Ajuste del parámetro $\beta$ durante el entrenamiento", fontsize=14, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
ruta_grafico3 = carpeta + "graficosSIR/evolucion_beta.png"
fig3.savefig(ruta_grafico3)

# ========================
# Gráfico 4: Susceptibles 
# ========================
fig4, ax2 = plt.subplots(figsize=(16, 8), dpi=300)

ax2.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3, label='Zona de Entrenamiento')
ax2.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_TOT, color='lightyellow', alpha=0.3, label='Zona de Test')
ax2.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle="--", linewidth=2)

ax2.plot(S_obs_norm_full * N, label="S Observado (Sintético)", color="pink", lw=2, alpha=0.8)
ax2.plot(S_long.cpu().numpy() * N, label="S Predicho (Modelo SIR)", color="purple", linestyle="--", lw=3)

ax2.set_xlabel("Días Transcurridos", fontsize=14)
configurar_eje_y_millones(ax2)
ax2.set_title("Susceptibles (Escala en Millones)", fontsize=18, fontweight='bold')
ax2.legend(fontsize=12, loc="upper right")
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
ruta_grafico4 = carpeta + "graficosSIR/susceptibles_millones.png"
fig4.savefig(ruta_grafico4)


print(f"\nGráficos guardados exitosamente en:\n- {ruta_grafico1}\n- {ruta_grafico2}\n- {ruta_grafico3}\n- {ruta_grafico4}")

plt.show()


