import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
import os

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
carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

# Crear carpeta para gráficos si no existe
carpeta_graficos = os.path.join(carpeta, "SIR_UDE")
if not os.path.exists(carpeta_graficos):
    os.makedirs(carpeta_graficos)

#==================================
#Carga y Preprocesamiento de datos
#==================================
df = pd.read_csv(CSV_PATH, usecols=["fecha_apertura", "clasificacion_resumen"])

#Filtramos solo confirmados para evitar ruido
df = df[df["clasificacion_resumen"] == "Confirmado"]

DATE_COLUMN = "fecha_apertura"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

#Contamos casos diarios
full_range = pd.date_range(start=df[DATE_COLUMN].min(), end=df[DATE_COLUMN].max(), freq='D')
cases_per_day = df.groupby(DATE_COLUMN).size().reindex(full_range, fill_value=0).sort_index()

#Suavizamos para eliminar el efecto del fin de semana
cases_per_day = cases_per_day.rolling(7, center=True).mean().bfill().ffill()

nuevos_casos_diarios = cases_per_day.values.astype(np.float32)

#Población total aproximada de Argentina
N = 46000000

#Cantidad de días que vamos a usar para entrenar
MAX_DAYS_TRAIN = 300
MAX_DAYS_VAL=30
MAX_DAYS_TOT=len(nuevos_casos_diarios)

#Casos de train
train_cases=nuevos_casos_diarios[:MAX_DAYS_TRAIN]
test_cases=nuevos_casos_diarios[MAX_DAYS_TRAIN:MAX_DAYS_TOT]

#===========================
#RECONSTRUCCIÓN SINTÉTICA
#===========================

#Asumimos un tiempo medio de recuperación de 14 días (parámetro clínico)
tau_random = int(np.random.normal(loc=14, scale=2))
tau_random = max(1, tau_random)  #evitamos valores negativos o cero

# Inicializamos vectores para las trayectorias
I_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)
R_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)
S_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)

I_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)
R_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)
S_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)

#Reconstrucción día por día de la cantidad total de infectados activos y recuperados
for t in range(MAX_DAYS_TRAIN):
    #Los infectados activos hoy son la suma de los que se contagiaron en los últimos TAU días
    inicio_ventana = max(0, t - tau_random + 1)
    I_sintetico_train[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])
    I_sintetico_full[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])

    #Los recuperados hoy son todos los que se contagiaron antes de esa ventana
    if t >= tau_random:
        R_sintetico_train[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])
        R_sintetico_full[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])

for t in range(MAX_DAYS_TRAIN, MAX_DAYS_TOT):
    #Para los días de test, seguimos la misma lógica pero con los datos completos
    inicio_ventana = max(0, t - tau_random + 1)
    I_sintetico_full[t] = np.sum(nuevos_casos_diarios[inicio_ventana : t + 1])
    if t >= tau_random:
        R_sintetico_full[t] = np.sum(nuevos_casos_diarios[:inicio_ventana])

# Evitar división por cero si max == min
I_obs_norm_full = I_sintetico_full / N
R_obs_norm_full = R_sintetico_full / N
S_obs_norm_full = 1.0 - I_obs_norm_full - R_obs_norm_full

# Separar train de la misma curvas full ya normalizadas
S_obs_norm_train = S_obs_norm_full[:MAX_DAYS_TRAIN]
I_obs_norm_train = I_obs_norm_full[:MAX_DAYS_TRAIN]
R_obs_norm_train = R_obs_norm_full[:MAX_DAYS_TRAIN]

#Tensor de tiempo
t_train_torch = torch.arange(MAX_DAYS_TRAIN, dtype=torch.float32, device=DEVICE)
t_val_torch = torch.arange(MAX_DAYS_TRAIN, MAX_DAYS_TRAIN + MAX_DAYS_VAL, dtype=torch.float32, device=DEVICE)
t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)
t_train_val_torch = torch.arange(MAX_DAYS_TRAIN + MAX_DAYS_VAL, dtype=torch.float32, device=DEVICE)


S_obs_torch = torch.tensor(S_obs_norm_train, dtype=torch.float32, device=DEVICE)
I_obs_torch = torch.tensor(I_obs_norm_train, dtype=torch.float32, device=DEVICE)
R_obs_torch = torch.tensor(R_obs_norm_train, dtype=torch.float32, device=DEVICE)

S_full_torch = torch.tensor(S_obs_norm_full, dtype=torch.float32, device=DEVICE)
I_full_torch = torch.tensor(I_obs_norm_full, dtype=torch.float32, device=DEVICE)
R_full_torch = torch.tensor(R_obs_norm_full, dtype=torch.float32, device=DEVICE)

# 2. Vector de tiempo que cubre TODO el rango de datos reales disponibles

# Agrupamos las tres variables en la matriz real de entrenamiento (Julia)
y_real_full = torch.stack([S_full_torch, I_full_torch, R_full_torch]).T
y_real_train = y_real_full[:MAX_DAYS_TRAIN]
y_real_val = y_real_full[MAX_DAYS_TRAIN:MAX_DAYS_TRAIN + MAX_DAYS_VAL]
y_real_test = y_real_full[MAX_DAYS_TRAIN + MAX_DAYS_VAL:MAX_DAYS_TOT]

inp_mean = y_real_train.mean(dim=0)
inp_std = y_real_train.std(dim=0).clamp(min=1e-5) + 1e-8
 
#u0 = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
#u0 = y_real_train[0].clone().detach()
u0 = y_real_train[0].clone().detach().to(torch.float32).to(DEVICE)

tspan = torch.linspace(0., float(MAX_DAYS_TRAIN-1), steps=MAX_DAYS_TRAIN, dtype=torch.float32, device=DEVICE)
sol_true = y_real_full

#target = sol_true.detach()
target = y_real_train.detach()

I_real_acumulado = np.cumsum(I_sintetico_full).astype(np.float32)

#=================================================================
# DETECCIÓN CAUSAL DE INICIOS DE OLA (usa SOLO datos pasados)
#=================================================================
# En el día t solo se usan datos de días <= t (diferencias hacia atrás),
# coherente con un escenario de tiempo real.

def detectar_inicios_olas(serie_diaria, ventana=10, sep_min=50,
                          crecimiento_rel=0.01, dias_confirmacion=5,
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
    I_sintetico_full.astype(np.float64)
)
# El día 0 siempre es punto de arranque; sumamos los inicios de ola.

print(f"Olas confirmadas en el día (tiempo real): {inicios_detectados}")
print(f"Reinicio anclado al valle (retroactivo): {inicios_olas}")

segmentos = [0] + [int(t) for t in inicios_olas if MAX_DAYS_TRAIN < t < MAX_DAYS_TOT]

#==========================================================
#  RECONSTRUCCIÓN DEL ESTADO REAL EN UN DÍA (solo observables)
#==========================================================
def estado_inicial_en(t_idx):

    vent = 14
    ini = max(0, t_idx - vent)
    I_t = I_obs_norm_full[t_idx]
    R_t = R_obs_norm_full[t_idx]
    S_t = S_obs_norm_full[t_idx]
    return torch.tensor([S_t, I_t, R_t], dtype=torch.float32, device=DEVICE)

#==========================================================
# PREDICCIÓN POR SEGMENTOS (un único modelo, reinicio por ola)
#==========================================================
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
        sol = odeint(model, u0_seg, t_seg, method='dopri5', rtol=1e-6, atol=1e-8)
        trozos.append(sol)
    return torch.cat(trozos, dim=0)

#==================
# SIR UDE
#==================
class InfectionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t, u):
        t_norm = (t/MAX_DAYS_TOT).view(1)

        u_scaled = torch.stack([u[0], u[1]*10000, u[2]*100])

        x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)

        return torch.sigmoid(self.net(x))*0.5 + 0.01

nn_model = InfectionNN().to(DEVICE)

class SIR_UDE(nn.Module):
    
    def __init__(self, infection_nn, gamma = 1/tau_random):
        super().__init__()
        self.infection_nn = infection_nn
        self.gamma = gamma
    
    def forward(self, t, u):
        S, I, R = u[0], u[1], u[2]

        beta_t = self.infection_nn(t, u).view([])
    
        infection = beta_t * S * I

        recovery = (self.gamma * I).to(torch.float32).view([])

        dS = (-infection).view([])
        dI = (infection - recovery).view([])
        dR = recovery.view([])

        return torch.stack([dS, dI, dR])

#======================================
# Predicción y función de pérdida
#======================================
def predict(params=None):
    sol = odeint(ude_model, u0.to(torch.float32), tspan, method='dopri5', rtol=1e-6, atol=1e-8)
    return sol

def loss_fn(pred, true):
    
    pred_I = pred[:, 1] * 10000.0
    true_I = true[:, 1] * 10000.0
    
    pred_R = pred[:, 2] * 10000.0
    true_R = true[:, 2] * 10000.0
    
    loss_I = torch.mean((pred_I - true_I)**2)
    loss_R = torch.mean((pred_R - true_R)**2)
    
    # Le damos prioridad a los infectados activos
    return loss_I + 0.3 * loss_R

#===============================================
# Entrenamiento: Fase 1 (Adam), Fase 2 (BFGS)
#===============================================

segmentos_train = [s for s in segmentos if s < MAX_DAYS_TRAIN]

adam_iters = 700
bfgs_iters = 200
loss_history = []
best_loss = float('inf')
best_state=None

print(f"Fase 1: Adam ({adam_iters} iters)")

nn_model = InfectionNN().to(DEVICE)
ude_model = SIR_UDE(nn_model).to(DEVICE)
optimizer_adam = optim.Adam(list(nn_model.parameters()), lr=1e-2, weight_decay=1e-4)

eps = 1e-7

segmentos_train = [s for s in segmentos if s < MAX_DAYS_TRAIN]

for epoch in range(adam_iters):
    optimizer_adam.zero_grad()
    pred = predecir_por_segmentos(ude_model, segmentos_train, t_train_torch)
    
    loss = loss_fn(pred, target)
    loss_history.append(loss)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        
    optimizer_adam.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print("\nIniciando Fase 2: L-BFGS...")
optimizer_lbfgs = torch.optim.LBFGS(nn_model.parameters(), max_iter=200, history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    
    # Predicción por segmentos: reinicia en el inicio de cada ola
    pred_y = predecir_por_segmentos(ude_model, segmentos_train, t_train_torch)

    # Agrego clamp para prevenir errores negativos
    
    loss = loss_fn(pred_y, target)
    
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print("Entrenamiento completado.")

with torch.no_grad():
    # Evaluamos con segmentos, tal como se entrenó
    pred_final_segmentos = predecir_por_segmentos(ude_model, segmentos_train, t_train_torch)
    loss_final_entrenamiento = loss_fn(pred_final_segmentos, target)
    
    # Evaluamos de corrido (para ver qué tan bien extrapola sin ayuda)
    pred_final_corrido = predict()
    loss_final_corrido = loss_fn(pred_final_corrido, target)

print(f"✔ Entrenamiento completado.")
print(f"-> Loss final real (por segmentos): {loss_final_entrenamiento.item():.6f}")
print(f"-> Loss de corrido (simulación libre): {loss_final_corrido.item():.6f}")

#========================
# PREPARACIÓN DATOS TEST
#========================
# Tensor de tiempo para toda la serie (Train + Test)
t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)

#========================
# GRAFICOS
#========================
casos_acumulados_reales = np.cumsum(nuevos_casos_diarios).astype(np.float32)

millions_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}')

def add_train_test_split(ax, train_days, max_dias):
    ax.axvline(x=train_days, color='black', linestyle='--', lw=1.5, label='Fin Training')
    ax.axvspan(0, train_days, alpha=0.05, color='blue', label='Fase Train') 
    ax.axvspan(train_days, max_dias, alpha=0.05, color='orange', label='Fase Test') 
    # El bucle de las olas se mantiene opcional
    for s in segmentos:
        if 0 < s < max_dias:
            ax.axvline(x=s, color='teal', linestyle=':', lw=1.2, alpha=0.8)

# Función para configurar etiquetas comunes y evitar repetir código
def apply_labels(ax, title, ylabel, xlabel="Días"):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

def add_train_test_split(ax, train_days, max_dias):
    ax.axvline(x=train_days, color='black', linestyle='--', lw=1.5, label='Fin Training')
    ax.axvspan(0, train_days, alpha=0.05, color='blue', label='Fase Train') 
    ax.axvspan(train_days, max_dias, alpha=0.05, color='orange', label='Fase Test') 
    # El bucle de las olas se mantiene opcional
    for s in segmentos:
        if 0 < s < max_dias:
            ax.axvline(x=s, color='teal', linestyle=':', lw=1.2, alpha=0.8)
def save_and_show(fig, filename):
    path = os.path.join(carpeta_graficos, f"{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {path}")
    plt.show()

def plot_sir_ude_results(model, y0, t_span, casos_acumulados_reales, nuevos_casos_diarios, I_sintetico_full, train_days):
    with torch.no_grad():
        segmentos_full = [s for s in segmentos if s < len(t_span)]
        pred_y = predecir_por_segmentos(model, segmentos_full, t_span)
        S, I, R = pred_y[:, 0].cpu().numpy(), pred_y[:, 1].cpu().numpy(), pred_y[:, 2].cpu().numpy()
        
        beta_learned = np.array([nn_model(t_span[i], pred_y[i]).detach().cpu().item() for i in range(len(t_span))])

    t_dias = t_span.cpu().numpy()
    max_dias = len(t_dias)
    nuevos_casos_pred = beta_learned * S * I * N
    casos_acumulados_pred = (S[0] - S) * N

    # Gráficos Individuales
    
    # 1. Beta (Agregando split)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_dias, beta_learned, color='darkred', lw=2, label=r"$\beta(t)$")
    add_train_test_split(ax1, train_days, max_dias)
    apply_labels(ax1, "Evolución de la Tasa de Transmisión", r"Valor de $\beta$")
    ax1.legend(loc="upper right")
    save_and_show(fig1, "01_beta")

    # 2. Infectados Activos (Agregando split)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_dias, I * N, label="Predicción UDE", color="orange", lw=2)
    ax2.plot(t_dias, I_sintetico_full[:max_dias], 'o', label="Datos Reales", color="black", alpha=0.2, markersize=3)
    ax2.yaxis.set_major_formatter(millions_formatter)
    add_train_test_split(ax2, train_days, max_dias)
    apply_labels(ax2, "Infectados Activos (I)", "Millones de personas")
    ax2.legend(loc="upper right")
    save_and_show(fig2, "02_infectados_activos")

    # 3. Acumulados (Agregando split)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(t_dias, casos_acumulados_reales[:max_dias], 'o', label="Datos Reales", color="black", alpha=0.4, markersize=4)
    ax3.plot(t_dias, casos_acumulados_pred, label="Predicción UDE", color="crimson", lw=2)
    ax3.yaxis.set_major_formatter(millions_formatter)
    add_train_test_split(ax3, train_days, max_dias)
    apply_labels(ax3, "Casos Totales Acumulados", "Millones de personas")
    ax3.legend(loc="upper left")
    save_and_show(fig3, "03_acumulados")

    # 4. Incidencia (Agregando split)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(t_dias, nuevos_casos_diarios[:max_dias], 'o', label="Datos Reales", color="black", alpha=0.4, markersize=4)
    ax4.plot(t_dias, nuevos_casos_pred, label="Predicción UDE", color="red", lw=2)
    ax4.yaxis.set_major_formatter(millions_formatter)
    add_train_test_split(ax4, train_days, max_dias)
    apply_labels(ax4, "Nuevos Casos Diarios", "Millones de personas")
    ax4.legend(loc="upper right")
    save_and_show(fig4, "04_incidencia")

# Función Loss
def plot_loss(loss_history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([l.item() for l in loss_history], color='navy')
    ax.set_yscale('log')
    ax.set_title("Evolución de la Loss")
    save_and_show(fig, "05_loss")

# Ejecución
plot_sir_ude_results(ude_model, u0, t_tot_torch, casos_acumulados_reales, nuevos_casos_diarios, I_sintetico_full, MAX_DAYS_TRAIN)
plot_loss(loss_history)