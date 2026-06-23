import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.optimize import minimize
from scipy.signal import savgol_filter
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

save_folder = os.path.join(carpeta, "SIRD_UDE")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=========================================================================
# Carga y preprocesamiento de datos reales (OPTIMIZADO POR CHUNKS)
#=========================================================================
chunks = []
# Leemos el archivo en fragmentos de 100.000 filas
for chunk in pd.read_csv(
    CSV_PATH, 
    usecols=["fecha_apertura", "fecha_fallecimiento", "clasificacion_resumen"],
    low_memory=False, 
    chunksize=100000
):
    # Filtramos por confirmados dentro de cada bloque
    chunk_filtrado = chunk[chunk["clasificacion_resumen"] == "Confirmado"].copy()
    chunks.append(chunk_filtrado)

# Concatenamos las filas limpias y confirmadas
df = pd.concat(chunks, ignore_index=True)
del chunks  # Liberamos y limpiamos los residuos

# Parseamos las fechas sobre el DataFrame reducido
df["fecha_apertura"] = pd.to_datetime(df["fecha_apertura"])
df["fecha_fallecimiento"] = pd.to_datetime(df["fecha_fallecimiento"])

DATE_COLUMN = "fecha_apertura"

# Casos y fallecidos diarios del dataset
casos_diarios_reales = df.groupby(DATE_COLUMN).size().sort_index()
fallecidos_diarios_reales = df[df["fecha_fallecimiento"].notna()].groupby("fecha_fallecimiento").size().sort_index()

# Alineamos índices cronológicos
all_dates = casos_diarios_reales.index
fallecidos_diarios_reales = fallecidos_diarios_reales.reindex(all_dates, fill_value=0)

# Suavizado de 7 días para limpiar el ruido de carga de datos
casos_diarios_suave = casos_diarios_reales.rolling(7, min_periods=1).mean().bfill().ffill()
fallecidos_diarios_suave = fallecidos_diarios_reales.rolling(7, min_periods=1).mean().bfill().ffill()

# Curvas acumuladas reales
D_real_acumulado = np.cumsum(fallecidos_diarios_suave).values.astype(np.float32)
C_real_acumulado = np.cumsum(casos_diarios_suave).values.astype(np.float32)
fallecidos_diarios_numpy = fallecidos_diarios_suave.values.astype(np.float32)

# Población total de Argentina
N = 46000000
MAX_DAYS_TRAIN = 300
MAX_DAYS_TOT = len(D_real_acumulado)

# Normalización respecto a N
D_obs_norm_train = D_real_acumulado[:MAX_DAYS_TRAIN] / N
dD_obs_norm_train = fallecidos_diarios_numpy[:MAX_DAYS_TRAIN] / N

# Conversión a Tensor de PyTorch
t_obs_torch = torch.tensor(np.arange(MAX_DAYS_TRAIN, dtype=np.float32), device=DEVICE)
D_obs_torch = torch.tensor(D_obs_norm_train, device=DEVICE)
dD_obs_torch = torch.tensor(dD_obs_norm_train, device=DEVICE)

# Tensores de tiempo para el entrenamiento
t_train_torch = torch.arange(MAX_DAYS_TRAIN, dtype=torch.float32, device=DEVICE)

#=================================================================
# DETECCIÓN CAUSAL DE INICIOS DE OLA (usando solo datos pasados)
#=================================================================

def detectar_inicios_olas(serie_diaria, ventana=10, sep_min=50,
                          crecimiento_rel=0.015, dias_confirmacion=5,
                          frac_pico_min=0.05, suavizado_valle = 15):

    x = np.asarray(serie_diaria, dtype=np.float64)
    n = len(x)
    d1 = np.zeros(n)

    for t in range(n):
        ini = max(0, t - ventana)
        if t - ini >= 2:
            tt = np.arange(ini, t + 1)
            d1[t] = np.polyfit(tt, x[ini:t + 1], 1)[0]

    inicios_detectados = []
    inicios_anclados = []
    ult = -sep_min
    racha = 0

    for t in range(ventana, n):
        nivel = max(x[t], 1.0)
        pico_hist = max(np.max(x[:t + 1]), 1.0)   # máximo solo hasta hoy

        crece_hoy = d1[t] > crecimiento_rel * nivel
        relevante = nivel > frac_pico_min * pico_hist or nivel < 0.5 * pico_hist

        if crece_hoy and relevante:
            racha += 1
        else:
            racha = 0

        if racha >= dias_confirmacion and (t - ult) >= sep_min:
            # Ola confirmada hoy. Buscamos el valle ya pasado:
            # mínimo en la ventana que precede a la racha actual.
            w = max(0, t - dias_confirmacion - ventana)
            subserie = x[w:t+1]
            subserie_suave=savgol_filter(subserie, suavizado_valle, polyorder=2, mode="interp")
            valle_rel = int(np.argmin(subserie_suave))
            valle = w + valle_rel
            
            inicios_detectados.append(t)
            inicios_anclados.append(valle)
            ult = t
            racha = 0

    return inicios_detectados, inicios_anclados, d1

inicios_detectados, inicios_olas, d1_f = detectar_inicios_olas(
    casos_diarios_suave.values.astype(np.float64)
)

print(f"Olas confirmadas en el día (tiempo real): {inicios_detectados}")
print(f"Reinicio anclado al valle (retroactivo): {inicios_olas}")

# El día 0 siempre es punto de arranque, sumamos los inicios de ola
segmentos = [0] + [int(t) for t in inicios_olas if MAX_DAYS_TRAIN < t < MAX_DAYS_TOT]

# Reconstrucción del estado real en cierto día
def estado_inicial_en(t_idx):
    D_t = D_real_acumulado[t_idx] / N
    C_t = C_real_acumulado[t_idx] / N
    vent = 14
    ini = max(0, t_idx - vent)
    I_t = (C_real_acumulado[t_idx] - C_real_acumulado[ini]) / N
    I_t = max(I_t, 10.0 / N)
    R_t = max(C_t - I_t - D_t, 0.0)
    S_t = max(1.0 - I_t - R_t - D_t, 0.0)
    return torch.tensor([S_t, I_t, R_t, D_t], dtype=torch.float32, device=DEVICE)

# Predicción por segmentos (reinicio del modelo por ola)
def predecir_por_segmentos(model, segmentos, t_full):
    n_tot = len(t_full)
    bordes = list(segmentos) + [n_tot]
    trozos = []
    for k in range(len(segmentos)):
        a, b = bordes[k], bordes[k + 1]
        if b <= a:
            continue
        u0_seg = estado_inicial_en(a)
        t_seg = torch.arange(a, b, dtype=torch.float32, device=DEVICE)
        sol = odeint(model, u0_seg, t_seg, method='dopri5')
        trozos.append(sol)
    return torch.cat(trozos, dim=0)

#=========================================================================
#SIRD UDE
#=========================================================================

# Definición de la Red Neuronal para aprender Beta
class InfectionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), # Entrada: [S, I, R, D]
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: beta_dinamico
        )

    def forward(self, t, u):
        # Normalizamos el tiempo respecto al máximo de días de train
        t_norm = (t / 300.0).view(1)
        
        # Escala artificial para ayudar a los gradientes de la red neuronal
        # S se mantiene igual, las demás se escalan para estar más cerca de 1.0
        u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
        
        x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
        
        # Limitamos la salida para que beta no explote ni sea negativom(entre 0.01 y 0.51)
        return torch.sigmoid(self.net(x)) + 0.01

# Definición del modelo SIRD UDE
class SIRD_UDE(nn.Module):
    def __init__(self, infection_nn, gamma_r=0.05, gamma_d=0.001):
        super().__init__()
        self.infection_nn = infection_nn
        self.gamma_r = gamma_r  # Tasa de recuperación
        self.gamma_d = gamma_d  # Tasa de mortalidad

    def forward(self, t, u):
        S, I, R, D = u[0], u[1], u[2], u[3]
        
        # La red neuronal aprende beta(t, S, I, R, D)
        beta_t = self.infection_nn(t, u).view([])
        
        infection = beta_t * S * I
        recovery = self.gamma_r * I
        mortality = self.gamma_d * I
        
        dS = -infection
        dI = infection - (recovery + mortality)
        dR = recovery
        dD = mortality
        
        return torch.stack([dS, dI, dR, dD])
    
#=========================================================================
# CONDICIONES INICIALES Y PREPARACION PARA EL ENTRENAMIENTO
#=========================================================================
# Estimación del Día 0
D0 = D_obs_norm_train[0]
C0 = C_real_acumulado[0] / N

I0 = C0 - D0 if (C0 - D0) > 0 else 10.0 / N 
R0 = 0.0
S0 = 1.0 - I0 - R0 - D0

u0 = torch.tensor([S0, I0, R0, D0], dtype=torch.float32, device=DEVICE)

infection_nn = InfectionNN().to(DEVICE)
model_ude = SIRD_UDE(infection_nn).to(DEVICE)

# Optimizador Fase 1
optimizer_adam = torch.optim.Adam(model_ude.parameters(), lr=1e-3)

#=========================================================================
# ENTRENAMIENTO
#=========================================================================
eps = 1e-7
loss_history = []

segmentos_train = [s for s in segmentos if s < MAX_DAYS_TRAIN]

print("Iniciando Fase 1: Adam")
for epoch in range(500): 
    optimizer_adam.zero_grad()

    # Predicción por segmentos: reinicia en el inicio de cada ola
    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    # Agregamos clamp para prevenir errores negativos
    D_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dD_pred = torch.clamp(model_ude.gamma_d * pred_y[:, 1], min=0.0)
    
    # Pérdida logarítmica enfocada en D y dD
    loss = torch.mean((torch.log(D_pred + eps) - torch.log(D_obs_torch + eps))**2) + \
           100*torch.mean((torch.log(dD_pred + eps) - torch.log(dD_obs_torch + eps))**2)
    
    loss.backward()
    optimizer_adam.step()

    loss_history.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print("\nIniciando Fase 2: L-BFGS")
optimizer_lbfgs = torch.optim.LBFGS(model_ude.parameters(), max_iter=200, history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    
    # Predicción por segmentos: reinicia en el inicio de cada ola
    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    # Agregamos clamp para prevenir errores negativos
    D_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dD_pred = torch.clamp(model_ude.gamma_d * pred_y[:, 1], min=0.0)
    
    loss = torch.mean((torch.log(D_pred + eps) - torch.log(D_obs_torch + eps))**2) + \
           100*torch.mean((torch.log(dD_pred + eps) - torch.log(dD_obs_torch + eps))**2)
    
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print("Entrenamiento completado.")

# Tensor de tiempo para toda la serie (Train + Test)
t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)

#=========================================================================
# GRAFICOS
#=========================================================================

def plot_and_save_sird_ude_results(model, y0, t_span, fallecidos_acum_real, fallecidos_diarios_real, train_days, losses, folder):
    # Asegurar que el directorio de salida existe
    if not os.path.exists(folder):
        os.makedirs(folder)

    # --- GRÁFICO 1: Historial de Pérdida (Loss Curve) ---
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(losses, color='royalblue', lw=2, label='Función de Pérdida total')
    plt.yscale('log')
    plt.title('Evolución de la Pérdida (Loss) durante el Entrenamiento', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Evaluaciones / Épocas', fontsize=12)
    plt.ylabel('Pérdida (Escala Logarítmica)', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '01_loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Obtención de variables latentes y predicciones del modelo
    with torch.no_grad():
        segmentos_full = [s for s in segmentos if s < len(t_span)]
        pred_y = predecir_por_segmentos(model, segmentos_full, t_span)

        S = pred_y[:, 0].cpu().numpy()
        I = pred_y[:, 1].cpu().numpy()
        R = pred_y[:, 2].cpu().numpy()
        D = pred_y[:, 3].cpu().numpy()
        
        beta_learned = []
        for i in range(len(t_span)):
            t_val = t_span[i]
            u_val = pred_y[i]
            beta_val = model.infection_nn(t_val, u_val).detach().cpu().item()
            beta_learned.append(beta_val)
        beta_learned = np.array(beta_learned)

    t_dias = t_span.cpu().numpy()
    max_dias = len(t_dias)
    fallecidos_acum_real = fallecidos_acum_real[:max_dias]
    fallecidos_diarios_real = fallecidos_diarios_real[:max_dias]
    fallecidos_diarios_pred = (model.gamma_d * I) * N

    # Función auxiliar para unificar la estética de Train/Test Split
    def aplicar_formato_train_test(ax, titulo, ylabel):
        ax.set_title(titulo, fontsize=14, pad=15, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Días", fontsize=12)
        
        # Línea de división Train / Test nítida y clara
        ax.axvline(x=train_days, color='black', linestyle='--', lw=2.5, zorder=5, label='Límite Train/Test')
        # Sombreados de fondo sutiles para identificar las regiones
        ax.axvspan(0, train_days, alpha=0.06, color='blue', label='Zona Entrenamiento (Train)')
        ax.axvspan(train_days, max_dias, alpha=0.06, color='orange', label='Zona Predicción (Test)')
        
        # Inicios de olas detectadas mapeadas como líneas de puntos
        primera_ola = True
        for s in segmentos:
            if 0 < s < max_dias:
                if primera_ola:
                    ax.axvline(x=s, color='darkcyan', linestyle=':', lw=1.5, alpha=0.8, zorder=4, label='Reinicio por Ola')
                    primera_ola = False
                else:
                    ax.axvline(x=s, color='darkcyan', linestyle=':', lw=1.5, alpha=0.8, zorder=4)
                    
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, loc='upper left')

    # --- GRÁFICO 2: Beta en el tiempo ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, beta_learned, color='firebrick', lw=2.5, label=r'$\beta(t)$ dinámico aprendido')
    aplicar_formato_train_test(ax, r"Evolución Temporal del Parámetro de Transmisión $\beta(t)$", r"Valor de $\beta$")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '02_evolucion_beta.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 3: Compartimentos SIRD ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, S, label="Susceptibles (S)", color="forestgreen", lw=2.5)
    ax.plot(t_dias, I, label="Infectados Activos (I)", color="darkorange", lw=2.5)
    ax.plot(t_dias, R, label="Recuperados (R)", color="mediumpurple", lw=2.5)
    aplicar_formato_train_test(ax, "Dinámica de Compartimentos Poblacionales Latentes (SIRD)", "Proporción de la Población")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '03_compartimentos_sird.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 4: Predicho vs Real - Fallecidos Acumulados ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, fallecidos_acum_real, 'o', label="Datos Reales", color="black", alpha=0.3, markersize=3.5)
    ax.plot(t_dias, D * N, label="Predicción UDE (D)", color="crimson", lw=2.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
    aplicar_formato_train_test(ax, "Ajuste y Extrapolación: Fallecidos Acumulados", "Cantidad de Personas")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '04_fallecidos_acumulados.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 5: Predicho vs Real - Fallecidos Diarios ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, fallecidos_diarios_real, 'o', label="Datos Reales (Suavizados)", color="black", alpha=0.3, markersize=3.5)
    ax.plot(t_dias, fallecidos_diarios_pred, label="Predicción UDE", color="red", lw=2.5)
    aplicar_formato_train_test(ax, "Ajuste y Extrapolación: Nuevos Fallecimientos Diarios", "Fallecimientos por Día")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, '05_fallecidos_diarios.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n¡Todos los gráficos se han guardado con éxito en la carpeta: '{folder}'!")

# Ejecutar proceso automático de guardado e impresión de gráficos en alta definición
plot_and_save_sird_ude_results(
    model_ude, u0, t_tot_torch, D_real_acumulado, 
    fallecidos_diarios_numpy, MAX_DAYS_TRAIN, loss_history, save_folder
)