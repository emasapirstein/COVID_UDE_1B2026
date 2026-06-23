import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.ticker as ticker
import os


#=========================================================================
# Semillas
#=========================================================================
def set_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(123)

#=========================================================================
# Configuración general
#=========================================================================
DEVICE = "cpu"
carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

save_folder = os.path.join(carpeta, "SIRD")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=========================================================================
# Carga y preprocesamiento de datos reales (OPTIMIZADO POR CHUNKS)
#=========================================================================
chunks = []
# Leemos el archivo en fragmentos de 100.000 filas para evitar OOM Crashes
for chunk in pd.read_csv(
    CSV_PATH, 
    usecols=["fecha_apertura", "fecha_fallecimiento", "clasificacion_resumen"],
    low_memory=False, 
    chunksize=100000
):
    # Filtramos por confirmados dentro de cada bloque
    chunk_filtrado = chunk[chunk["clasificacion_resumen"] == "Confirmado"].copy()
    chunks.append(chunk_filtrado)

# Concatenamos únicamente las filas filtradas y confirmadas
df = pd.concat(chunks, ignore_index=True)
del chunks  # Liberamos y limpiamos los residuos 

# Parseamos las fechas únicamente sobre el DataFrame reducido final
df["fecha_apertura"] = pd.to_datetime(df["fecha_apertura"])
df["fecha_fallecimiento"] = pd.to_datetime(df["fecha_fallecimiento"])

DATE_COLUMN = "fecha_apertura"

# Casos diarios y fallecidos diarios del dataset
casos_diarios_reales = df.groupby(DATE_COLUMN).size().sort_index()
fallecidos_diarios_reales = df[df["fecha_fallecimiento"].notna()].groupby("fecha_fallecimiento").size().sort_index()

# Alineamos índices cronológicos
all_dates = casos_diarios_reales.index
fallecidos_diarios_reales = fallecidos_diarios_reales.reindex(all_dates, fill_value=0)

# Suavizado de 7 días para limpiar el ruido de carga de datos
casos_diarios_suave = casos_diarios_reales.rolling(7, center=True).mean().bfill().ffill()
fallecidos_diarios_suave = fallecidos_diarios_reales.rolling(7, center=True).mean().bfill().ffill()

# Curvas acumuladas reales directas
D_real_acumulado = np.cumsum(fallecidos_diarios_suave).values.astype(np.float32)
C_real_acumulado = np.cumsum(casos_diarios_suave).values.astype(np.float32)

# Población total de Argentina
N = 46000000

MAX_DAYS_TRAIN = 300
MAX_DAYS_TOT = len(D_real_acumulado)

# Normalización respecto a N
D_obs_norm_train = D_real_acumulado[:MAX_DAYS_TRAIN] / N
D_obs_norm_full = D_real_acumulado / N
C_obs_norm_full = C_real_acumulado / N

# Conversión a Tensor de PyTorch
t_obs_torch = torch.tensor(np.arange(MAX_DAYS_TRAIN, dtype=np.float32), device=DEVICE)
D_obs_torch = torch.tensor(D_obs_norm_train, device=DEVICE)

#=========================================================================
# MODELO SIRD PURO CLÁSICO
#=========================================================================
class PureSIRDModel(nn.Module):
    def __init__(self, beta=0.25, gamma_r=0.05, gamma_d=0.001):
        super().__init__()
        # Parámetros libres que optimizará Adam basándose SOLO en la curva D
        self.beta_raw = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.gamma_r_raw = nn.Parameter(torch.tensor(gamma_r, dtype=torch.float32))
        self.gamma_d_raw = nn.Parameter(torch.tensor(gamma_d, dtype=torch.float32))

    def forward(self, t, u):
        S, I, R, D = u

        # Filtro softplus para asegurar positividad de las tasas
        beta = torch.nn.functional.softplus(self.beta_raw)
        gamma_r = torch.nn.functional.softplus(self.gamma_r_raw)
        gamma_d = torch.nn.functional.softplus(self.gamma_d_raw)

        # Ecuaciones diferenciales del SIRD clásico
        infection = beta * S * I
        recovery = gamma_r * I
        mortality = gamma_d * I

        dS = -infection
        dI = infection - (recovery + mortality)
        dR = recovery
        dD = mortality

        return torch.stack([dS, dI, dR, dD])

#=========================================================================
# CONDICIONES INICIALES REALES / ESTIMADAS
#=========================================================================

D0 = D_obs_norm_train[0]
C0 = C_obs_norm_full[0] # Usamos el inicio de casos acumulados para fijar un I0 coherente

# Inicialización realista del Día 0
I0 = C0 - D0 if (C0 - D0) > 0 else 10.0 / N 
R0 = 0.0
S0 = 1.0 - I0 - R0 - D0

y0 = torch.tensor([S0, I0, R0, D0], dtype=torch.float32, device=DEVICE)

model = PureSIRDModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

#=========================================================================
# ENTRENAMIENTO SOBRE LA VARIABLE DE FALLECIDOS (D)
#=========================================================================
n_epochs = 3000
loss_history = []
beta_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Resolvemos el sistema SIRD de 4 variables
    pred_y = odeint(model, y0, t_obs_torch, method='rk4')

    D_pred = pred_y[:, 3]

    # Pérdida logarítmica enfocada UNICAMENTE en la curva de muertes
    loss = torch.mean((torch.log(D_pred + 1e-6) - torch.log(D_obs_torch + 1e-6)) ** 2)

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # Recuperamos el valor de beta para graficar
    beta_actual = torch.nn.functional.softplus(model.beta_raw).item()
    beta_history.append(beta_actual)

    if epoch % 300 == 0:
        beta_val = beta_actual
        g_r_val = torch.nn.functional.softplus(model.gamma_r_raw).item()
        g_d_val = torch.nn.functional.softplus(model.gamma_d_raw).item()
        print(f"Epoch {epoch:4d} | Loss (Solo D): {loss.item():.6f} | beta: {beta_val:.4f} | gamma_r: {g_r_val:.4f} | gamma_d: {g_d_val:.4f}")

#=========================================================================
# SIMULACIÓN E INFERENCIA A LARGO PLAZO
#=========================================================================
MAX_DAYS_SIM = 800
t_sim = torch.arange(MAX_DAYS_SIM, dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    pred_y_long = odeint(model, y0, t_sim, method='rk4')

S_long, I_long, R_long, D_long = [pred_y_long[:, i].cpu().numpy() for i in range(4)]

#=========================================================================
# VISUALIZACIÓN DE RESULTADOS REALES VS LATENTES DEDUCIDOS
#=========================================================================

# --- GRÁFICO 1: EVOLUCIÓN DEL LOSS ---
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='purple', lw=2.5)
plt.yscale('log')
plt.title('Evolución de la Función de Pérdida (Loss)', fontsize=13, fontweight='bold')
plt.xlabel('Épocas', fontsize=11)
plt.ylabel('Log-Loss', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "1_evolucion_loss.png"), dpi=300, bbox_inches='tight')

# --- GRÁFICO 2: EVOLUCIÓN DE BETA ---
plt.figure(figsize=(10, 6))
plt.plot(beta_history, color='teal', lw=2.5)
plt.title('Convergencia de $\\beta$ durante el Entrenamiento', fontsize=13, fontweight='bold')
plt.xlabel('Épocas', fontsize=11)
plt.ylabel('Valor de $\\beta$', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "2_evolucion_beta.png"), dpi=300, bbox_inches='tight')

# --- GRÁFICO 3: FALLECIDOS ACUMULADOS (PREDICHO VS REAL) ---
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3, label='Zona de Train')
ax1.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_SIM, color='lightgoldenrodyellow', alpha=0.3, label='Zona de Test (Inferencia)')
plt.plot(D_obs_norm_full * N, label="Fallecidos Reales (Dataset)", color="black", lw=3)
plt.plot(D_long * N, label="Fallecidos (Predicción SIRD)", color="crimson", linestyle="--", lw=3)
plt.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle=":", lw=2, label=f"Línea de Corte (Día {MAX_DAYS_TRAIN})")
plt.ylabel("Cantidad de Personas", fontsize=11)
plt.xlabel("Días desde el inicio", fontsize=11)
plt.title("Ajuste de Fallecidos Acumulados (D) - Entrenamiento vs Inferencia", fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "3_fallecidos_acumulados.png"), dpi=300, bbox_inches='tight')

# --- GRÁFICO 4: VALIDACIÓN CRUZADA DE CASOS ACUMULADOS ---
fig, ax2 = plt.subplots(figsize=(14, 6))
ax2.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3, label='Zona de Train')
ax2.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_SIM, color='lightgoldenrodyellow', alpha=0.3, label='Zona de Test (Inferencia)')
plt.plot(C_obs_norm_full * N, label="Casos Confirmados Reales", color="darkblue", alpha=0.7, lw=3)
plt.plot((1.0 - S_long) * N, label="Casos Teóricos Deducidos (1 - S)", color="dodgerblue", linestyle="--", lw=3)
plt.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle=":", lw=2, label=f"Línea de Corte (Día {MAX_DAYS_TRAIN})")
plt.ylabel("Cantidad de Personas", fontsize=11)
plt.xlabel("Días desde el inicio", fontsize=11)
plt.title("Validación Cruzada: Casos Acumulados Reales vs. Curva Deducida", fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "4_casos_acumulados_validacion.png"), dpi=300, bbox_inches='tight')

# --- GRÁFICO 5: COMPARTIMENTOS LATENTES LIBRES ---
fig, ax3 = plt.subplots(figsize=(14, 6))
ax3.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3, label='Zona de Train')
ax3.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_SIM, color='lightgoldenrodyellow', alpha=0.3, label='Zona de Test (Inferencia)')
plt.plot(I_long * N, label="Infectados Activos (Latente)", color="orange", lw=3)
plt.plot(R_long * N, label="Recuperados (Latente)", color="green", lw=3)
plt.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle=":", lw=2, label=f"Línea de Corte (Día {MAX_DAYS_TRAIN})")
plt.ylabel("Cantidad de Personas", fontsize=11)
plt.xlabel("Días desde el inicio", fontsize=11)
plt.title("Evolución de los Compartimentos Latentes Libres (I, R)", fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "5_compartimentos_latentes.png"), dpi=300, bbox_inches='tight')
plt.show()