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
CSV_PATH = os.path.join(carpeta, "Covid19Casos.csv")

ruta_salida_completa = os.path.join(carpeta, "graficosSIRX")
os.makedirs(ruta_salida_completa, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: No encuentro el archivo en: {CSV_PATH}")
else:
    print(f"ÉXITO: Archivo encontrado en {CSV_PATH}. Procediendo con el entrenamiento")

#=========================================================================
# Carga y Preprocesamiento de datos 
#=========================================================================

# Nos quedamos solo con las columnas fecha_apertura y clasificacion_resumen
# Filtramos solo casos confirmados
df = pd.read_csv(CSV_PATH, usecols=["fecha_apertura", "clasificacion_resumen"])
df = df[df["clasificacion_resumen"] == "Confirmado"]

# Pasamos fecha_apertura a tipo datetime
DATE_COLUMN = "fecha_apertura"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

# Contamos casos nuevos diarios y suavizamos
cases_per_day = df.groupby(DATE_COLUMN).size().sort_index()
cases_per_day = cases_per_day.rolling(7, center=True).mean().bfill().ffill()

# Paso a array para convertir en Tensor
nuevos_casos_diarios_real = cases_per_day.values.astype(np.float32)

# Casos acumulados totales X
casos_acumulados_real = cases_per_day.cumsum().values.astype(np.float32)

# Poblacion total aproximada de Argentina
N = 46000000

# Cantidad de dias que vamos a usar para entrenar
# Cantidad total de dias
MAX_DAYS_TRAIN = 300
MAX_DAYS_TOT = len(nuevos_casos_diarios_real)

# Normalizamos los casos diarios respecto a la población N
casos_diarios_norm_train = nuevos_casos_diarios_real[:MAX_DAYS_TRAIN] / N
casos_diarios_norm_full = nuevos_casos_diarios_real / N

# Normalizamos los casos acumulados 
casos_acum_norm_train = casos_acumulados_real[:MAX_DAYS_TRAIN] / N
casos_acum_norm_full = casos_acumulados_real / N

# Vector de tiempo y conversión a Tensores
t_obs = np.arange(MAX_DAYS_TRAIN, dtype=np.float32)
t_obs_torch = torch.tensor(t_obs, device=DEVICE)

# Convertimos los casos diarios reales normalizados en tensor para la loss
casos_diarios_real_torch = torch.tensor(casos_diarios_norm_train, device=DEVICE)
casos_acum_real_torch = torch.tensor(casos_acum_norm_train, device=DEVICE)

#=========================================================================
# Modelo SIRX
#=========================================================================
class SIRXModel(nn.Module):
    # inicialización y Parametros
    def __init__(self, beta=0.902, kappa=0.03125):
        super().__init__()
        # Parámetros libres que optimizará Adam (Tasa de transmisión inicial y kIX)
        self.beta_raw = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.kappa_raw = nn.Parameter(torch.tensor(kappa, dtype=torch.float32)) # kIX en el paper 
        self.gamma = torch.tensor(0.125, dtype=torch.float32) # según el paper: gamma = 1/8 = 0.125 (no se optimiza) 

    # Ecuaciones diferenciales
    def forward(self, t, u):
        S, I, R, X = u

        # Garantizamos positividad usando softplus
        beta = torch.nn.functional.softplus(self.beta_raw)
        kappa = torch.nn.functional.softplus(self.kappa_raw)

        # Flujos epidemiológicos
        infection = beta * S * I
        recovery = self.gamma * I
        detection_flux = kappa * I     

        dS = -infection
        dI = infection - recovery- detection_flux
        dR = recovery 
        dX = detection_flux 

        return torch.stack([dS, dI, dR, dX])

#=========================================================================
# Condiciones iniciales 
#=========================================================================

# Proporcion inicial de Infectados Confirmados Activos
X0 = casos_acum_norm_train[0]
# Proporcion inicial de Infectados Activos Ocultos (pocos)
I0 = 100.0 / N            
# Proporción basada en el ratio inicial de R     
R0 = X0 * (0.125 / 0.03)  
# Proporcion inical de suceptibles
S0 = 1.0 - I0 - R0 - X0

# Condiciones iniciales en objeto Tensor
y0 = torch.tensor([S0, I0, R0, X0], dtype=torch.float32, device=DEVICE)

# Modelo SIRX
model = SIRXModel().to(DEVICE)
# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

#=========================================================================
# Entrenamiento
#=========================================================================

# Epochs y guardado de error
n_epochs = 3000
loss_history = []
beta_history = []
eps = 1e-7

for epoch in range(n_epochs):
    # Reseteamos a cero los gradientes acumulados
    optimizer.zero_grad()

    # Resolvemos ODE para SIRX
    pred_y = odeint(model, y0, t_obs_torch, method='rk4')

    # Casos acumulados predichos
    X_pred = pred_y[:,3] 

    # Casos diarios predichos
    kappa_actual = torch.nn.functional.softplus(model.kappa_raw) # kIX
    dX_pred = kappa_actual * pred_y[:,1]

    # Funcion de perdida
    # Penalizacion logaritmica
    loss_acumulados = torch.mean((torch.log(X_pred + eps) - torch.log(casos_acum_real_torch + eps))**2)
    loss_diarios = torch.mean((torch.log(dX_pred + eps)- torch.log(casos_diarios_real_torch + eps))**2)

    # Suma de la loss
    loss = loss_acumulados + loss_diarios

    # Backpropagation y optimizacion
    loss.backward()
    optimizer.step()

    # Registro de la loss 
    loss_history.append(loss.item())

    beta_actual = torch.nn.functional.softplus(model.beta_raw).item()
    beta_history.append(beta_actual)

    # Informe Epidemiológico 
    if epoch % 300 == 0:
        beta = beta_actual
        kappa = torch.nn.functional.softplus(model.kappa_raw).item()
        
        # Calculamos la probabilidad de aislamiento epidemiológico
        # gamma está fijo en 0.125 según las pautas del documento 
        q_prob = kappa / (kappa + 0.125) 
        
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | beta (transmisión): {beta:.4f} | kappa (kIX): {kappa:.4f} | Qprob (Detección): {q_prob*100:.2f}%")

#=========================================================================
# Inferencia a largo plazo
#=========================================================================
MAX_DAYS_SIM = 800
t_sim = torch.arange(MAX_DAYS_SIM, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    pred_y_long = odeint(model, y0, t_sim, method='rk4')

S_long, I_long, R_long, X_long = pred_y_long[:,0], pred_y_long[:,1], pred_y_long[:,2], pred_y_long[:,3]

kappa_final = torch.nn.functional.softplus(model.kappa_raw).item()
nuevos_casos_diarios_pred = (kappa_final * I_long.detach().cpu().numpy()) * N

# Conversión a escala poblacional para gráficos
X_acumulado_pred_plot = X_long.detach().cpu().numpy() * N
I_activo_calle_plot = I_long.detach().cpu().numpy() * N

# =====================================================================
# Visualización de Resultados Actualizados
# =====================================================================

S_pred_plot = S_long.detach().cpu().numpy() * N
I_pred_plot = I_long.detach().cpu().numpy() * N
R_pred_plot = R_long.detach().cpu().numpy() * N
X_pred_plot = X_long.detach().cpu().numpy() * N

# Función de guardado
def guardar_grafico(fig, nombre):
    ruta_archivo = os.path.join(ruta_salida_completa, f"{nombre}.png")
    fig.savefig(ruta_archivo, dpi=400, bbox_inches='tight')
    print(f"Gráfico guardado: {nombre}.png")
    plt.close(fig) # Cierra la figura para no frenar el script

# Formateador para el eje Y (Millones)
millions_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}')
def configurar_eje_y_millones(ax):
    ax.set_ylabel("Millones de Personas", fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(millions_formatter)

# --- GRÁFICO 1: Predicho vs Real (Acumulados) con Train/Test ---

fig1, ax1 = plt.subplots(figsize=(16, 8))
# Zonas de color de fondo
ax1.axvspan(0, MAX_DAYS_TRAIN, color='#e6f2ff', alpha=0.5, label='Entrenamiento')
ax1.axvspan(MAX_DAYS_TRAIN, MAX_DAYS_TOT, color='#fff9e6', alpha=0.5, label='Test/Predicción')
# Línea divisoria
ax1.axvline(x=MAX_DAYS_TRAIN, color="red", linestyle="--", linewidth=2.5, 
            label="Límite Train/Test")

# Datos
ax1.plot(casos_acumulados_real, label="Real (Datos Oficiales)", color="black", lw=3, alpha=0.7)
ax1.plot(X_pred_plot, label="Predicho (Modelo SIRX)", color="#0055ff", lw=3, linestyle="--")

ax1.set_xlabel("Días Transcurridos", fontsize=14, fontweight='bold')
configurar_eje_y_millones(ax1)
ax1.set_title("Casos Acumulados (X): Ajuste Real vs Predicción", fontsize=18, fontweight='bold')
ax1.legend(fontsize=14, loc="lower right", framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.7, color='gray')
ax1.tick_params(axis='both', which='major', labelsize=12)

guardar_grafico(fig1, "01_Ajuste_Real_vs_Predicho")


# --- GRÁFICO 2: Evolución de la Loss ---
fig2, ax2 = plt.subplots(figsize=(14, 7))

ax2.plot(loss_history, color="#800080", lw=2.5)
ax2.set_yscale('log') # Escala logarítmica fundamental para ver la convergencia
ax2.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax2.set_ylabel("MSE Loss (Escala Log)", fontsize=14, fontweight='bold')
ax2.set_title("Curva de Aprendizaje (Convergencia de la Loss)", fontsize=18, fontweight='bold')
ax2.grid(True, linestyle='-', alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=12)

guardar_grafico(fig2, "02_Evolucion_Loss")


# --- GRÁFICO 3: Evolución de Beta en el tiempo (Optimizador) ---
fig3, ax3 = plt.subplots(figsize=(14, 7))

ax3.plot(beta_history, color="#008080", lw=2.5, label=r"Tasa de transmisión ($\beta$)")
# Línea horizontal con el valor final
beta_final = beta_history[-1]
ax3.axhline(y=beta_final, color="red", linestyle=":", lw=2, label=f"Valor Final: {beta_final:.4f}")

ax3.set_xlabel("Epochs", fontsize=14, fontweight='bold')
ax3.set_ylabel("Valor del Parámetro", fontsize=14, fontweight='bold')
ax3.set_title("Evolución del parámetro Beta durante el entrenamiento", fontsize=18, fontweight='bold')
ax3.legend(fontsize=14)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.tick_params(axis='both', which='major', labelsize=12)

guardar_grafico(fig3, "03_Evolucion_Beta")


# --- GRÁFICO 4: Dinámica Epidemiológica - Solo Susceptibles (S) ---

fig4, ax4 = plt.subplots(figsize=(16, 8))

# Zonas de color y límite
ax4.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3)
ax4.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle="--", linewidth=2)

# Curva S
ax4.plot(S_pred_plot, label="S (Susceptibles)", color="#2ca02c", lw=3)

ax4.set_xlabel("Días Transcurridos", fontsize=14, fontweight='bold')
configurar_eje_y_millones(ax4)
ax4.set_title("Dinámica Epidemiológica: Población Susceptible (S)", fontsize=18, fontweight='bold')
ax4.legend(fontsize=14, loc="upper right", framealpha=0.9)
ax4.grid(True, linestyle=':', alpha=0.7)
ax4.tick_params(axis='both', which='major', labelsize=12)

guardar_grafico(fig4, "04_Dinamica_Susceptibles")


# --- GRÁFICO 5: Dinámica Epidemiológica - I, R, X ---

fig5, ax5 = plt.subplots(figsize=(16, 8))

# Zonas de color y límite
ax5.axvspan(0, MAX_DAYS_TRAIN, color='lightgray', alpha=0.3)
ax5.axvline(x=MAX_DAYS_TRAIN, color="black", linestyle="--", linewidth=2)

# Curvas I, R, X
ax5.plot(I_pred_plot, label="I (Infectados Activos Ocultos)", color="#ff7f0e", lw=2.5)
ax5.plot(R_pred_plot, label="R (Removidos)", color="#9467bd", lw=2.5)
ax5.plot(X_pred_plot, label="X (Confirmados Acumulados)", color="#1f77b4", lw=2.5, linestyle="-.")

ax5.set_xlabel("Días Transcurridos", fontsize=14, fontweight='bold')
configurar_eje_y_millones(ax5) # Usamos la misma escala en millones para consistencia
ax5.set_title("Dinámica Epidemiológica: I, R, X (Sin Susceptibles)", fontsize=18, fontweight='bold')
ax5.legend(fontsize=14, loc="upper left", framealpha=0.9)
ax5.grid(True, linestyle=':', alpha=0.7)
ax5.tick_params(axis='both', which='major', labelsize=12)

guardar_grafico(fig5, "05_Dinamica_IRX")
plt.show()