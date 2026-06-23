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

#=========================================================================
#Configuración general
#=========================================================================
DEVICE = "cpu"
carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

save_folder = os.path.join(carpeta, "SIRD_VARIABLES")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=========================================================================
# Carga y Preprocesamiento de Datos Reales (OPTIMIZADO POR CHUNKS)
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
del chunks  # Liberamos y limpiamos los residuos de la memoria RAM al instante

# Parseamos las fechas sobre el DataFrame reducido
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

#=========================================================================
# DETECCIÓN CAUSAL DE INICIOS DE OLA (usando solo datos pasados)
#=========================================================================

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

# El día 0 siempre es punto de arranquen sumamos los inicios de ola.
segmentos = [0] + [int(t) for t in inicios_olas if MAX_DAYS_TRAIN < t < MAX_DAYS_TOT]

# Reconstrucción del estado real en cierto día
def estado_inicial_en(t_idx):
    D_t = D_real_acumulado[t_idx] / N
    C_t = C_real_acumulado[t_idx] / N
    vent = 14
    ini = max(0, t_idx - vent)
    I_t = (C_real_acumulado[t_idx] - C_real_acumulado[ini]) / N
    I_t = max(I_t, 100.0 / N)  
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

class BetaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  
        )

    def forward(self, t, u):
        if u.dim() == 1:
            # Caso 1: Un solo paso (dentro de odeint)
            t_norm = (t / MAX_DAYS_TOT).view(1)
            u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
            x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
            return torch.sigmoid(self.net(x)) * 0.5 + 0.01 
        else:
            # Caso 2: Lote completo (cálculo de loss fuera de odeint)
            t_norm = (t / MAX_DAYS_TOT).view(-1, 1)
            u_scaled = torch.stack([u[:, 0], u[:, 1]*1000, u[:, 2]*10, u[:, 3]*1000], dim=1)
            x = torch.cat([t_norm, u_scaled], dim=1)
            return (torch.sigmoid(self.net(x)) * 0.5 + 0.01).view(-1) 
        

class Gamma_r_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  
        )

    def forward(self, t, u):
        if u.dim() == 1:
            t_norm = (t / MAX_DAYS_TOT).view(1)
            u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
            x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
            return torch.sigmoid(self.net(x)) * 0.1 + 0.01
        else:
            t_norm = (t / MAX_DAYS_TOT).view(-1, 1)
            u_scaled = torch.stack([u[:, 0], u[:, 1]*1000, u[:, 2]*10, u[:, 3]*1000], dim=1)
            x = torch.cat([t_norm, u_scaled], dim=1)
            return (torch.sigmoid(self.net(x)) * 0.1 + 0.01).view(-1)


class Gamma_d_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  
        )

    def forward(self, t, u):
        if u.dim() == 1:
            t_norm = (t / MAX_DAYS_TOT).view(1)
            u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
            x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
            return torch.sigmoid(self.net(x))*0.01 + 0.001
        else:
            t_norm = (t / MAX_DAYS_TOT).view(-1, 1)
            u_scaled = torch.stack([u[:, 0], u[:, 1]*1000, u[:, 2]*10, u[:, 3]*1000], dim=1)
            x = torch.cat([t_norm, u_scaled], dim=1)
            return (torch.sigmoid(self.net(x))*0.01 + 0.001).view(-1)
        

# Definición del modelo SIRD UDE
class SIRD_UDE(nn.Module):
    def __init__(self, beta_nn, gamma_r_nn, gamma_d_nn):
        super().__init__()
        self.beta = beta_nn # Tasa de contagio
        self.gamma_r = gamma_r_nn  # Tasa de recuperación
        self.gamma_d = gamma_d_nn  # Tasa de mortalidad

    def forward(self, t, u):
        S, I, R, D = u[0], u[1], u[2], u[3]
        
        beta_t = self.beta(t, u).view([])
        gamma_r_t = self.gamma_r(t, u).view([])
        gamma_d_t = self.gamma_d(t, u).view([])
        
        infection = beta_t * S * I
        recovery = gamma_r_t * I
        mortality = gamma_d_t * I
        
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

beta_nn = BetaNN().to(DEVICE)
gamma_r_nn = Gamma_r_NN().to(DEVICE)
gamma_d_nn = Gamma_d_NN().to(DEVICE)

model_ude = SIRD_UDE(beta_nn, gamma_r_nn, gamma_d_nn).to(DEVICE)

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

    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    # Agregamos clamp para prevenir errores negativos
    D_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dD_pred = torch.clamp(model_ude.gamma_d(t_train_torch, pred_y) * pred_y[:, 1], min=0.0)
    
    D_scale = torch.mean(D_obs_torch ** 2) + 1e-6
    dD_scale = torch.mean(dD_obs_torch ** 2) + 1e-6

    loss = torch.mean((D_pred - D_obs_torch) ** 2) / D_scale + \
       100 * torch.mean((dD_pred - dD_obs_torch) ** 2) / dD_scale
    
    loss.backward()
    loss_history.append(loss.item())
    optimizer_adam.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print("\nIniciando Fase 2: L-BFGS")
optimizer_lbfgs = torch.optim.LBFGS(model_ude.parameters(), max_iter=200, history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    
    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    # Agregamos clamp para prevenir errores negativos
    D_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dD_pred = torch.clamp(model_ude.gamma_d(t_train_torch, pred_y) * pred_y[:, 1], min=0.0)
    
    D_scale = torch.mean(D_obs_torch ** 2) + 1e-6
    dD_scale = torch.mean(dD_obs_torch ** 2) + 1e-6

    loss = torch.mean((D_pred - D_obs_torch) ** 2) / D_scale + \
       100 * torch.mean((dD_pred - dD_obs_torch) ** 2) / dD_scale
    
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print("Entrenamiento completado.")

# Tensor de tiempo para toda la serie (Train + Test)
t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)

#=========================================================================
# GRAFICOS
#=========================================================================
def plot_sird_ude_results_separados(model, y0, t_span, fallecidos_acum_real, fallecidos_diarios_real, train_days, loss_hist, save_folder):
    with torch.no_grad():
        # Predecimos sobre TODA la serie reiniciando en el inicio de cada ola.
        segmentos_full = [s for s in segmentos if s < len(t_span)]
        pred_y = predecir_por_segmentos(model, segmentos_full, t_span)

        S = pred_y[:, 0].cpu().numpy()
        I = pred_y[:, 1].cpu().numpy()
        R = pred_y[:, 2].cpu().numpy()
        D = pred_y[:, 3].cpu().numpy()
        
        beta_learned = []
        gamma_r_learned = []
        gamma_d_learned = []
        for i in range(len(t_span)):
            t_val = t_span[i]
            u_val = pred_y[i]
            beta_learned.append(model.beta(t_val, u_val).detach().cpu().item())
            gamma_r_learned.append(model.gamma_r(t_val, u_val).detach().cpu().item())
            gamma_d_learned.append(model.gamma_d(t_val, u_val).detach().cpu().item())
        
        beta_learned = np.array(beta_learned)
        gamma_r_learned = np.array(gamma_r_learned)
        gamma_d_learned = np.array(gamma_d_learned)

    t_dias = t_span.cpu().numpy()
    max_dias = len(t_dias)
    
    fallecidos_acum_real = fallecidos_acum_real[:max_dias]
    fallecidos_diarios_real = fallecidos_diarios_real[:max_dias]

    # Predicción diaria: dD = gamma_d * I * N
    fallecidos_diarios_pred = (gamma_d_learned * I) * N

    # Formatters para los ejes Y
    thousands_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e3:.0f}k')

    # Función auxiliar para aplicar el estilo de Train/Test y olas a cada gráfico
    def aplicar_formato_base(ax, titulo, ylabel, xlabel="Días", y_formatter=None, mostrar_leyenda=True):
        ax.axvline(x=train_days, color='#2c3e50', linestyle='--', lw=2.5, zorder=3, label='Límite Train/Test')
        ax.axvspan(0, train_days, alpha=0.06, color='#3498db', label='Zona Train') 
        ax.axvspan(train_days, max_dias, alpha=0.06, color='#e74c3c', label='Zona Test')
        
        for s in segmentos:
            if 0 < s < max_dias:
                ax.axvline(x=s, color='teal', linestyle=':', lw=1.5, alpha=0.7, zorder=3)
        ax.plot([], [], color='teal', linestyle=':', lw=1.5, label='Inicio de ola')
        
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if y_formatter:
            ax.yaxis.set_major_formatter(y_formatter)
        if mostrar_leyenda:
            ax.legend(loc='best', fontsize=11, framealpha=0.9)

    os.makedirs(save_folder, exist_ok=True)

    # 1. GRÁFICO: Evolución de la Función de Pérdida
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_hist, color='#8e44ad', lw=2.5, label='Loss (Fase 1)')
    plt.title("Evolución de la Función de Pérdida", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Loss (Escala Log)", fontsize=12)
    plt.xlabel("Épocas", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    path_loss = os.path.join(save_folder, "01_loss_evolution.png")
    plt.savefig(path_loss, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. GRÁFICO: Parámetros Aprendidos (Beta, Gamma_R y Gamma_D)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, beta_learned, color='#c0392b', lw=2.5, label=r"$\beta(t)$ (Infección)")
    ax.plot(t_dias, gamma_r_learned, color='#27ae60', lw=2.5, label=r"$\gamma_r(t)$ (Recuperación)")
    ax.plot(t_dias, gamma_d_learned, color='#2980b9', lw=2.5, label=r"$\gamma_d(t)$ (Mortalidad)")
    aplicar_formato_base(ax, "Evolución Temporal de Parámetros Aprendidos", "Valor del parámetro")
    path_params = os.path.join(save_folder, "02_learned_parameters.png")
    plt.savefig(path_params, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. GRÁFICO: Compartimentos Latentes (S, I, R)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, S, label="Susceptibles (S)", color="#27ae60", lw=2.5)
    ax.plot(t_dias, I, label="Infectados Activos (I)", color="#d35400", lw=2.5)
    ax.plot(t_dias, R, label="Recuperados (R)", color="#8e44ad", lw=2.5)
    # Sin formatter de miles acá porque graficamos proporciones de 0.0 a 1.0
    aplicar_formato_base(ax, "Dinámica de Compartimentos Latentes", "Proporción de la Población")
    path_compartimentos = os.path.join(save_folder, "03_latent_compartments.png")
    plt.savefig(path_compartimentos, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. GRÁFICO: Fallecidos Acumulados
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.scatter(t_dias, fallecidos_acum_real, color='black', alpha=0.3, s=12, label="Datos Reales Acumulados")
    ax.plot(t_dias, D * N, color='#c0392b', lw=2.5, label="Predicción UDE (D)")
    aplicar_formato_base(ax, "Fallecidos Totales Acumulados", "Personas", y_formatter=thousands_formatter)
    path_acumulados = os.path.join(save_folder, "04_cumulative_deaths.png")
    plt.savefig(path_acumulados, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. GRÁFICO: Nuevos Fallecidos Diarios (Incidencia)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.scatter(t_dias, fallecidos_diarios_real, color='black', alpha=0.3, s=15, label="Datos Reales Diarios (Suavizados)")
    ax.plot(t_dias, fallecidos_diarios_pred, color='#e74c3c', lw=2.5, label="Predicción UDE (Nuevos/Día)")
    # Quitamos el formatter de miles porque los diarios no suelen superar los 1000 y se verían como "0k"
    aplicar_formato_base(ax, "Incidencia: Fallecimientos Diarios", "Fallecimientos por Día")
    path_diarios = os.path.join(save_folder, "05_daily_deaths.png")
    plt.savefig(path_diarios, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Todos los gráficos fueron exportados exitosamente en '{save_folder}':")
    print("   -> 01_loss_evolution.png")
    print("   -> 02_learned_parameters.png")
    print("   -> 03_latent_compartments.png")
    print("   -> 04_cumulative_deaths.png")
    print("   -> 05_daily_deaths.png")

# Ejecución de la exportación por archivos separados
plot_sird_ude_results_separados(
    model_ude, 
    u0, 
    t_tot_torch, 
    D_real_acumulado, 
    fallecidos_diarios_numpy, 
    MAX_DAYS_TRAIN,
    loss_history,     
    save_folder       
)