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

save_folder = os.path.join(carpeta, "SIRX_VARIABLES")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=========================================================================
# Carga y Preprocesamiento de datos 
#=========================================================================

# Nos quedamos solo con las columnas fecha_apertura y clasificacion_resumen
#Filtramos solo confirmados
df = pd.read_csv(CSV_PATH, usecols=["fecha_apertura", "clasificacion_resumen"])
df = df[df["clasificacion_resumen"] == "Confirmado"]

# Pasamos fecha_apertura a tipo datetime
DATE_COLUMN = "fecha_apertura"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

# Contamos casos nuevos diarios y suavizamos
cases_per_day = df.groupby(DATE_COLUMN).size().sort_index()
cases_per_day = cases_per_day.rolling(7, center=True).mean().bfill().ffill()

# Paso a array para convertir despues en Tensor
nuevos_casos_diarios_real = cases_per_day.astype(np.float32)

# Casos acumulados totales 
casos_acumulados_real = cases_per_day.cumsum().astype(np.float32)

# Parametros Epidemiologicos y Poblacion total aproximada de Argentina
N = 46000000
GAMMA = 0.05
KAPPA = 0.01125

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

# Tensores de observación (Entrenamiento)
t_train_torch = torch.arange(MAX_DAYS_TRAIN, dtype=torch.float32, device=DEVICE)
t_full_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)
casos_diarios_real_torch = torch.tensor(casos_diarios_norm_train.to_numpy(), dtype=torch.float32, device=DEVICE)
casos_acum_real_torch = torch.tensor(casos_acum_norm_train.to_numpy(), dtype=torch.float32, device=DEVICE)

# Estimación simple de variables no observadas (S, I, R)
I_obs_full = (nuevos_casos_diarios_real / N) / KAPPA
R_obs_full = (casos_acum_norm_full * (GAMMA / KAPPA)) 
S_obs_full = 1.0 - I_obs_full - R_obs_full - casos_acum_norm_full

y_real_full = torch.stack([
    torch.tensor(S_obs_full.to_numpy(), device=DEVICE, dtype=torch.float32),
    torch.tensor(I_obs_full.to_numpy(), device=DEVICE, dtype=torch.float32),
    torch.tensor(R_obs_full.to_numpy(), device=DEVICE, dtype=torch.float32),
    torch.tensor(casos_acum_norm_full.to_numpy(), device=DEVICE, dtype=torch.float32)
]).T

# Condición Inicial (Día 0)
u0 = y_real_full[0].clone().detach().to(DEVICE)

# Detección de inicios de ola 
# En el día t solo se usan datos de días <= t (diferencias hacia atrás),
# la detección es coherente con un escenario de tiempo real.
def detectar_inicios_olas(serie_diaria, ventana=10, sep_min=50,
                          crecimiento_rel=0.015, dias_confirmacion=5,
                          frac_pico_min=0.05, suavizado_valle = 15):
    
    x = np.asarray(serie_diaria, dtype=np.float64)
    n = len(x)
    d1 = np.zeros(n)

    # Pendiente causal: ajuste lineal sobre [t-ventana, t]  (solo pasado)
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
        pico_hist = max(np.max(x[:t + 1]), 1.0)   # máximo SOLO hasta hoy

        crece_hoy = d1[t] > crecimiento_rel * nivel
        relevante = nivel > frac_pico_min * pico_hist or nivel < 0.5 * pico_hist

        if crece_hoy and relevante:
            racha += 1
        else:
            racha = 0

        if racha >= dias_confirmacion and (t - ult) >= sep_min:

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
    nuevos_casos_diarios_real.values.astype(np.float64)
)
# El día 0 siempre es punto de arranque; sumamos los inicios de ola.

print(f"Olas confirmadas en el día (tiempo real): {inicios_detectados}")
print(f"Reinicio anclado al valle (retroactivo): {inicios_olas}")

segmentos = [0] + [int(t) for t in inicios_olas if MAX_DAYS_TRAIN < t < MAX_DAYS_TOT]

# Estado inicial en cada ola

def estado_inicial_en(t_idx):
    C_t = casos_acum_norm_full.iloc[t_idx]
    vent = 14
    ini = max(0, t_idx - vent)
    I_t = (casos_acum_norm_full.iloc[t_idx] -casos_acum_norm_full.iloc[ini])
    I_t = max(I_t, 10.0 / N)
    R_t = max(C_t - I_t, 0.0)
    S_t = max(1.0 - I_t - R_t, 0.0)
    return torch.tensor([S_t, I_t, R_t, C_t], dtype=torch.float32, device=DEVICE)

# Predicción por segmentos

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
        sol = odeint(model, u0_seg, t_seg, method='rk4', options={'step_size': 0.5})
        trozos.append(sol)
    return torch.cat(trozos, dim=0)

#=========================================================================
#SIRX UDE
#=========================================================================
# Definicion de las redes neuronales

class BetaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), # Entrada: [t, S, I, R, X]
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: beta_dinamico
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
    
class GammaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), # Entrada: [t, S, I, R, X]
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: beta_dinamico
        )

    def forward(self, t, u):
        if u.dim() == 1:
            t_norm = (t / MAX_DAYS_TOT).view(1)
            u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
            x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
            return torch.sigmoid(self.net(x)) + 0.01
        else:
            t_norm = (t / MAX_DAYS_TOT).view(-1, 1)
            u_scaled = torch.stack([u[:, 0], u[:, 1]*1000, u[:, 2]*10, u[:, 3]*1000], dim=1)
            x = torch.cat([t_norm, u_scaled], dim=1)
            return (torch.sigmoid(self.net(x)) + 0.01).view(-1)

class KappaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), # Entrada: [t, S, I, R, X]
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: beta_dinamico
        )

    def forward(self, t, u):
        if u.dim() == 1:
            t_norm = (t / MAX_DAYS_TOT).view(1)
            u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
            x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
            return torch.sigmoid(self.net(x)) + 0.01
        else:
            t_norm = (t / MAX_DAYS_TOT).view(-1, 1)
            u_scaled = torch.stack([u[:, 0], u[:, 1]*1000, u[:, 2]*10, u[:, 3]*1000], dim=1)
            x = torch.cat([t_norm, u_scaled], dim=1)
            return (torch.sigmoid(self.net(x)) + 0.01).view(-1)

# Definicion de SIRX UDE
class SIRX_UDE(nn.Module):
    def __init__(self, beta_nn, gamma_nn, kappa_nn):
        super().__init__()
        self.beta = beta_nn
        self.gamma = gamma_nn  
        self.kappa = kappa_nn  

    def forward(self, t, u):
        u_safe = torch.clamp(u, min=0.0)
        S, I, R, X = u_safe[0], u_safe[1], u_safe[2], u_safe[3]
        
        beta_t = self.beta(t, u_safe).view([])
        gamma_t = self.gamma(t, u_safe).view([])
        kappa_t = self.kappa(t, u_safe).view([])
        
        infection = beta_t * S * I
        recovery = gamma_t * I
        detection = kappa_t * I
        
        dS = -infection
        dI = infection - recovery - detection
        dR = recovery
        dX = detection
        
        return torch.stack([dS, dI, dR, dX])


beta_nn = BetaNN().to(DEVICE)
gamma_nn = GammaNN().to(DEVICE)
kappa_nn = KappaNN().to(DEVICE)

# Modelo SIRX UDE

model_ude = SIRX_UDE(beta_nn, gamma_nn, kappa_nn).to(DEVICE)

t_span = t_full_torch

y_real_train = y_real_full[:MAX_DAYS_TRAIN]

u0 = y_real_train[0].clone().detach().to(DEVICE)
target = y_real_train.detach()

#=========================================================================
# ENTRENAMIENTO
#=========================================================================
eps = 1e-7
loss_history = []
optimizer_adam = torch.optim.Adam(model_ude.parameters(), lr=1e-4)

# Segmentos limitados al rango de entrenamiento. Un único modelo se entrena sobre todos los tramos

segmentos_train = [s for s in segmentos if s < MAX_DAYS_TRAIN]

print("Iniciando Fase 1: Adam")
for epoch in range(500): 
    optimizer_adam.zero_grad()

    # Predicción por segmentos: reinicia en el inicio de cada ola
    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    # Aseguramos que la solución no contenga NaNs o Infs
    if torch.isnan(pred_y).any() or torch.isinf(pred_y).any():
        print("Integración inestable, reduciendo learning rate...")
        optimizer_adam.param_groups[0]['lr'] *= 0.5

    # Agregamos clamp para prevenir errores negativos
    X_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dX_pred = torch.clamp(model_ude.kappa(t_train_torch, pred_y) * pred_y[:, 1], min=0.0)
    
    X_pred_personas = X_pred * N
    dX_pred_personas = dX_pred * N

    casos_acum_reales_personas = casos_acum_real_torch * N
    casos_diarios_reales_personas = casos_diarios_real_torch * N

    loss = torch.mean(((X_pred - casos_acum_real_torch) / (casos_acum_real_torch.max() + 1e-6))**2) + \
           torch.mean(((dX_pred - casos_diarios_real_torch) / (casos_diarios_real_torch.max() + 1e-6))**2)
    
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model_ude.parameters(), max_norm=0.5)

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
    X_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dX_pred = torch.clamp(model_ude.kappa(t_train_torch, pred_y) * pred_y[:, 1], min=0.0)
    
    X_pred_personas = X_pred * N
    dX_pred_personas = dX_pred * N

    casos_acum_reales_personas = casos_acum_real_torch * N
    casos_diarios_reales_personas = casos_diarios_real_torch * N

    loss = torch.mean(((X_pred - casos_acum_real_torch) / (casos_acum_real_torch.max() + 1e-6))**2) + \
           torch.mean(((dX_pred - casos_diarios_real_torch) / (casos_diarios_real_torch.max() + 1e-6))**2)
    
    
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print("Entrenamiento completado.")

# Tensor de tiempo para toda la serie (Train + Test)
t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)

# Convertimos los datos completos (Train + Test) para los gráficos
casos_acum_full_torch = torch.tensor(casos_acum_norm_full.to_numpy(), dtype=torch.float32, device=DEVICE)
casos_diarios_full_torch = torch.tensor(casos_diarios_norm_full.to_numpy(), dtype=torch.float32, device=DEVICE)

#=========================================================================
# GRAFICOS
#=========================================================================

def plot_sirx_ude_results_separados(model, t_span, sol_trained, casos_acum_full, casos_diarios_full, loss_hist, train_days, max_days, save_folder):
    S = sol_trained[:, 0].detach().cpu().numpy()
    I = sol_trained[:, 1].detach().cpu().numpy()
    R = sol_trained[:, 2].detach().cpu().numpy()
    X = sol_trained[:, 3].detach().cpu().numpy()
    
    beta_l = []
    gamma_l = []
    kappa_l = []
    for i in range(len(t_span)):
        t_val = t_span[i]
        u_val = sol_trained[i]
        beta_l.append(model.beta(t_val, u_val).detach().cpu().item())
        gamma_l.append(model.gamma(t_val, u_val).detach().cpu().item())
        kappa_l.append(model.kappa(t_val, u_val).detach().cpu().item())
        
    beta_l = np.array(beta_l)
    gamma_l = np.array(gamma_l)
    kappa_l = np.array(kappa_l)

    t_dias = np.arange(max_days)
    casos_diarios_pred = (kappa_l * I) * N
    
    millions_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')
    thousands_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e3:.0f}k')

    def aplicar_formato_base(ax, titulo, ylabel, xlabel="Días", y_formatter=None, mostrar_leyenda=True):
        ax.axvline(x=train_days, color='#2c3e50', linestyle='--', lw=2.5, zorder=3, label='Límite Train/Test')
        ax.axvspan(0, train_days, alpha=0.06, color='#3498db', label='Zona Train') 
        ax.axvspan(train_days, max_days, alpha=0.06, color='#e74c3c', label='Zona Test')
        
        for s in segmentos:
            if 0 < s < max_days:
                ax.axvline(x=s, color='teal', linestyle=':', lw=1.5, alpha=0.7, zorder=3)
        ax.plot([], [], color='teal', linestyle=':', lw=1.5, label='Inicio de ola')
        
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if y_formatter: ax.yaxis.set_major_formatter(y_formatter)
        if mostrar_leyenda: ax.legend(loc='best', fontsize=11, framealpha=0.9)

    os.makedirs(save_folder, exist_ok=True)

    # --- GRÁFICO 1: función de pérdida ---
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_hist, color='#8e44ad', lw=2.5)
    plt.yscale('log')
    plt.title("Evolución de la Función de Pérdida", fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_folder, "01_loss_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 2: parámetros aprendidos ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, beta_l, color='#c0392b', lw=2.5, label=r"$\beta(t)$ (Transmisión)")
    ax.plot(t_dias, gamma_l, color='#27ae60', lw=2.5, label=r"$\gamma(t)$ (Recuperación)")
    ax.plot(t_dias, kappa_l, color='#2980b9', lw=2.5, label=r"$\kappa(t)$ (Detección)")
    aplicar_formato_base(ax, "Evolución Temporal de Parámetros Aprendidos", "Valor del parámetro")
    plt.savefig(os.path.join(save_folder, "02_learned_parameters.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 3: compartimientos latentes ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, S, color="#27ae60", lw=2.5, label="Susceptibles (S)")
    ax.plot(t_dias, I, color="#d35400", lw=2.5, label="Infectados Activos (I)")
    ax.plot(t_dias, R, color="#8e44ad", lw=2.5, label="Recuperados (R)")
    aplicar_formato_base(ax, "Dinámica de Compartimentos Latentes", "Proporción de la Población")
    plt.savefig(os.path.join(save_folder, "03_latent_compartments.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 4: casos acumulados confirmados ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.scatter(t_dias, casos_acum_full.cpu().numpy() * N, color='black', alpha=0.3, s=12, label="Datos Reales")
    ax.plot(t_dias, X * N, color='#c0392b', lw=2.5, label="Predicción UDE (X)")
    aplicar_formato_base(ax, "Casos Confirmados Acumulados", "Personas", y_formatter=millions_formatter)
    plt.savefig(os.path.join(save_folder, "04_cumulative_confirmed.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 5: nuevos casos diarios ---
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.scatter(t_dias, casos_diarios_full.cpu().numpy() * N, color='black', alpha=0.3, s=15, label="Datos Reales Diarios")
    ax.plot(t_dias, casos_diarios_pred, color='#e74c3c', lw=2.5, label="Predicción UDE (Nuevos/Día)")
    aplicar_formato_base(ax, "Incidencia: Nuevos Casos Diarios Detectados", "Infectados por Día", y_formatter=thousands_formatter)
    plt.savefig(os.path.join(save_folder, "05_daily_cases.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Gráficos SIRX exportados exitosamente en '{save_folder}'")

# Ejecutamos la predicción y los gráficos sobre el rango completo
sol_final = predecir_por_segmentos(model_ude, segmentos_train, t_tot_torch)

plot_sirx_ude_results_separados(
    model=model_ude, 
    t_span=t_tot_torch, 
    sol_trained=sol_final, 
    casos_acum_full=casos_acum_full_torch, 
    casos_diarios_full=casos_diarios_full_torch, 
    loss_hist=loss_history, 
    train_days=MAX_DAYS_TRAIN, 
    max_days=MAX_DAYS_TOT, 
    save_folder=save_folder
)