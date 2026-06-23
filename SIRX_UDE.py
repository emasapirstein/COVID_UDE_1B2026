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
carpeta = carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

save_folder = os.path.join(carpeta, "SIRX_UDE")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=========================================================================
# Carga y Preprocesamiento de datos 
#=========================================================================
df = pd.read_csv(CSV_PATH, usecols=["fecha_apertura", "clasificacion_resumen"])
df = df[df["clasificacion_resumen"] == "Confirmado"]

DATE_COLUMN = "fecha_apertura"
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

cases_per_day = df.groupby(DATE_COLUMN).size().sort_index()
cases_per_day = cases_per_day.rolling(7, center=True).mean().bfill().ffill()

nuevos_casos_diarios_real = cases_per_day.astype(np.float32)
casos_acumulados_real = cases_per_day.cumsum().astype(np.float32)

N = 46000000
GAMMA = 0.05
KAPPA = 0.01125

MAX_DAYS_TRAIN = 300
MAX_DAYS_TOT = len(nuevos_casos_diarios_real)

casos_diarios_norm_train = nuevos_casos_diarios_real[:MAX_DAYS_TRAIN] / N
casos_diarios_norm_full = nuevos_casos_diarios_real / N

casos_acum_norm_train = casos_acumulados_real[:MAX_DAYS_TRAIN] / N
casos_acum_norm_full = casos_acumulados_real / N

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

u0 = y_real_full[0].clone().detach().to(DEVICE)

# Detección de inicios de olas

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
        pico_hist = max(np.max(x[:t + 1]), 1.0)

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

print(f"Olas confirmadas en el día (tiempo real): {inicios_detectados}")
print(f"Reinicio anclado al valle (retroactivo): {inicios_olas}")

segmentos = [0] + [int(t) for t in inicios_olas if MAX_DAYS_TRAIN < t < MAX_DAYS_TOT]

# Reconstrucción del estado inicial en un día

def estado_inicial_en(t_idx):
    C_t = casos_acum_norm_full.iloc[t_idx]
    vent = 14
    ini = max(0, t_idx - vent)
    I_t = (casos_acum_norm_full.iloc[t_idx] - casos_acum_norm_full.iloc[ini])
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
        sol = odeint(model, u0_seg, t_seg, method='dopri5')
        trozos.append(sol)
    return torch.cat(trozos, dim=0)

#=========================================================================
# SIRX UDE
#=========================================================================

class InfectionNN(nn.Module):
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
        t_norm = (t / 300.0).view(1)
        u_scaled = torch.stack([u[0], u[1]*1000, u[2]*10, u[3]*1000])
        x = torch.cat([t_norm, u_scaled], dim=0).unsqueeze(0)
        return torch.sigmoid(self.net(x)) + 0.01

class SIRX_UDE(nn.Module):
    def __init__(self, infection_nn, gamma=0.125, kappa=0.03125):
        super().__init__()
        self.infection_nn = infection_nn
        self.gamma = gamma  
        self.kappa = kappa  

    def forward(self, t, u):
        S, I, R, X = u[0], u[1], u[2], u[3]
        
        beta_t = self.infection_nn(t, u).view([])
        
        infection = beta_t * S * I
        recovery = self.gamma * I
        detection = self.kappa * I
        
        dS = -infection
        dI = infection - recovery - detection
        dR = recovery
        dX = detection
        
        return torch.stack([dS, dI, dR, dX])


infection_nn = InfectionNN().to(DEVICE)
model_ude = SIRX_UDE(infection_nn).to(DEVICE)

eps = 1e-7
t_span = t_full_torch
y_real_train = y_real_full[:MAX_DAYS_TRAIN]
u0 = y_real_train[0].clone().detach().to(DEVICE)
target = y_real_train.detach()

optimizer_adam = torch.optim.Adam(model_ude.parameters(), lr=1e-3)

#=========================================================================
# ENTRENAMIENTO
#=========================================================================
loss_history = []
segmentos_train = [s for s in segmentos if s < MAX_DAYS_TRAIN]

print("Iniciando Fase 1: Adam...")
for epoch in range(500): 
    optimizer_adam.zero_grad()

    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    X_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dX_pred = torch.clamp(model_ude.kappa * pred_y[:, 1], min=0.0)
    
    # Pérdida enfocada en X (Detectados Acumulados) y dX (Detectados Diarios)
    loss = torch.mean((torch.log(X_pred + eps) - torch.log(casos_acum_real_torch + eps))**2) + \
            50*torch.mean((torch.log(dX_pred + eps) - torch.log(casos_diarios_real_torch + eps))**2)
    
    loss.backward()
    optimizer_adam.step()
    
    loss_history.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print("\nIniciando Fase 2: L-BFGS...")
optimizer_lbfgs = torch.optim.LBFGS(model_ude.parameters(), max_iter=200, history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    
    pred_y = predecir_por_segmentos(model_ude, segmentos_train, t_train_torch)

    X_pred = torch.clamp(pred_y[:, 3], min=0.0)
    dX_pred = torch.clamp(model_ude.kappa * pred_y[:, 1], min=0.0)
    
    loss = torch.mean((torch.log(X_pred + eps) - torch.log(casos_acum_real_torch + eps))**2) + \
           100*torch.mean((torch.log(dX_pred + eps) - torch.log(casos_diarios_real_torch + eps))**2)
    
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print("Entrenamiento completado.")

t_tot_torch = torch.arange(MAX_DAYS_TOT, dtype=torch.float32, device=DEVICE)

#=========================================================================
# GRAFICOS 
#=========================================================================

def plot_and_save_all(model, t_span, casos_diarios, casos_acumulados, train_days, save_path):
    print("Generando y guardando gráficos en pantalla y disco")
    
    with torch.no_grad():
        segmentos_full = [s for s in segmentos if s < len(t_span)]
        pred_y = predecir_por_segmentos(model, segmentos_full, t_span)
        
        S = pred_y[:, 0].cpu().numpy()
        I = pred_y[:, 1].cpu().numpy()
        X = pred_y[:, 3].cpu().numpy()
        
        beta_learned = [model.infection_nn(t_span[i], pred_y[i]).item() for i in range(len(t_span))]

    # --- GRÁFICO 1: LOSS ---
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(loss_history, color='navy', lw=1.5)
    plt.yscale('log')
    plt.title("Evolución de la Función de Pérdida (Escala Log)")
    plt.xlabel("Épocas/Iteraciones")
    plt.ylabel("Loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(save_path, "01_Loss_Function.png"), bbox_inches='tight')
    plt.show() 
    plt.close()

    # --- GRÁFICO 2: RESULTADOS (Beta, Acumulados, Diarios) ---
    plot_configs = [
        (beta_learned, r"$\beta(t)$", "Evolucion_Tasa_Transmision_SIRX", r"$\beta$"),
        (X * N, "Predicción SIRX", "Casos_Acumulados_Detectados", "Personas"),
        ((model.kappa * I) * N, "Predicción SIRX", "Casos_Diarios_Detectados", "Casos/Día")
    ]
    
    real_data = [None, casos_acumulados, casos_diarios]

    for i, (pred, label_pred, title_str, ylabel) in enumerate(plot_configs):
        plt.figure(figsize=(12, 6), dpi=300)
        
        plt.axvspan(0, train_days, color='green', alpha=0.1, label='Zona de Entrenamiento')
        plt.axvspan(train_days, len(t_span), color='orange', alpha=0.1, label='Zona de Test')
        plt.axvline(x=train_days, color='black', linestyle='--', lw=2)
        
        if real_data[i] is not None:
            plt.plot(t_span.cpu(), real_data[i], 'o', color='gray', alpha=0.4, markersize=3, label="Datos Reales")
        plt.plot(t_span.cpu(), pred, color='crimson' if i > 0 else 'darkred', lw=2.5, label=label_pred)
        
        # Título limpio para visualizar en el plot
        display_title = title_str.replace("_", " ")
        plt.title(display_title)
        
        plt.ylabel(ylabel)
        plt.xlabel("Días")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardado con nombre limpio
        filename = f"0{i+2}_{title_str}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        plt.show() 
        plt.close()

# Llamada a la función
plot_and_save_all(model_ude, t_tot_torch, nuevos_casos_diarios_real.values, casos_acumulados_real.values, MAX_DAYS_TRAIN, save_folder)