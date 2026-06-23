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
from scipy.optimize import minimize
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


#=====================
#Configuración general
#=====================
DEVICE = "cpu"
carpeta = "Completar con la carpeta del CSV"
CSV_PATH = carpeta + "Covid19Casos.csv"

save_folder = os.path.join(carpeta, "SIRS_UDE")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Carpeta creada: {save_folder}")

#=====================
#Carga y Preprocesamiento de datos
#=====================
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
MAX_DAYS_TRAIN = 600
MAX_DAYS_VAL=30
MAX_DAYS_TOT=len(nuevos_casos_diarios)

#Casos de train
train_cases=nuevos_casos_diarios[:MAX_DAYS_TRAIN]
test_cases=nuevos_casos_diarios[MAX_DAYS_TRAIN:MAX_DAYS_TOT]

#=========================================================================
#RECONSTRUCCIÓN SINTÉTICA
#=========================================================================

#Asumimos un tiempo medio de recuperación de 14 días (parámetro clínico)
tau_random = int(np.random.normal(loc=14, scale=2))
tau_random = max(1, tau_random)  #evitamos valores negativos o cero

 # NUEVO: Asumimos un tiempo medio de inmunidad de 90 días (coherente con tu omega inicial de 1/90)
rho_random = max(30, int(np.random.normal(loc=90, scale=10)))
rho_random = max(1, rho_random) 

# Inicializamos vectores para las trayectorias
I_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)
R_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)
S_sintetico_train = np.zeros(MAX_DAYS_TRAIN, dtype=np.float32)

I_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)
R_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)
S_sintetico_full = np.zeros(MAX_DAYS_TOT, dtype=np.float32)

#Reconstrucción día por día de la cantidad total de infectados activos y recuperados
for t in range(MAX_DAYS_TOT):
    # Ventana de contagio activo [t - tau + 1, t]
    idx_start_I = max(0, t - tau_random + 1)
    I_sintetico_full[t] = np.sum(nuevos_casos_diarios[idx_start_I : t + 1])
    
    # Ventana de inmunidad [t - tau - rho + 1, t - tau]
    if t >= tau_random:
        idx_start_R = max(0, t - tau_random - rho_random + 1)
         # Sumamos los casos que ya no están infectados pero siguen siendo inmunes
        R_sintetico_full[t] = np.sum(nuevos_casos_diarios[idx_start_R : idx_start_I])

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

# Vector de tiempo que cubre TODO el rango de datos reales disponibles
# Agrupamos las tres variables en la matriz real de entrenamiento 
y_real_full = torch.stack([S_full_torch, I_full_torch, R_full_torch]).T
y_real_train = y_real_full[:MAX_DAYS_TRAIN]
y_real_val = y_real_full[MAX_DAYS_TRAIN:MAX_DAYS_TRAIN + MAX_DAYS_VAL]
y_real_test = y_real_full[MAX_DAYS_TRAIN + MAX_DAYS_VAL:MAX_DAYS_TOT]

inp_mean = y_real_train.mean(dim=0)
inp_std = y_real_train.std(dim=0).clamp(min=1e-2) + 1e-8
 
#u0 = torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
u0 = y_real_train[0].clone().detach()

tspan = torch.linspace(0., float(MAX_DAYS_TRAIN-1), steps=MAX_DAYS_TRAIN, dtype=torch.float32, device=DEVICE)
sol_true = y_real_full

#target = sol_true.detach()
target = y_real_train.detach()

# ---------------------------------------------------------------------------
# UDE: red neuronal para términos de interacción
# ---------------------------------------------------------------------------

class InfectionNN(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Inicialización personalizada 
        # Esto fuerza que el modelo no empiece con beta=0
        with torch.no_grad():
            self.net[-1].bias.fill_(0.2)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return torch.nn.functional.softplus(self.net(x))

nn_model = InfectionNN(inp_mean, inp_std).to(DEVICE)
log_tau= nn.Parameter(torch.tensor(np.log(14.0), dtype=torch.float32))
log_omega=nn.Parameter(torch.tensor(np.log(1/90), dtype = torch.float32))

def sir_ude(t, u):

    # Evitamos que tau tienda a 0 
    tau = torch.clamp(torch.exp(log_tau), min=1.0, max=100.0)
    gamma = 1.0 / tau
    
    # Evitamos que omega tome valores  absurdos
    omega = torch.clamp(torch.exp(log_omega), min=1e-5, max=1.0)
    
    S, I, R = u[0], u[1], u[2]
    

    # Si el solver pisa por un nanosegundo un valor negativo, torch.abs lo salva sin matar el flujo del gradiente como lo hace un clamp a 0.
    S_in, I_in, R_in = torch.abs(S), torch.abs(I), torch.abs(R)
    inp = torch.stack([S_in, I_in, R_in]).unsqueeze(0).to(torch.float32)
    
    # Limitamos la red para que no escupa una tasa irreal 
    infection_raw = torch.clamp(nn_model(inp), max=5.0).view([])

    # Calculamos los flujos estabilizados
    infection = infection_raw * S_in * I_in
    recovery = (gamma * I).to(torch.float32)
    warning = (omega * R).to(torch.float32)

    dS = (-infection + warning).view([])
    dI = (infection - recovery).view([])
    dR = (recovery - warning).view([])

    return torch.stack([dS, dI, dR])

# ---------------------------------------------------------------------------
# Predicción y función de pérdida
# ---------------------------------------------------------------------------
def predict(params=None):
    sol = odeint(sir_ude, u0.to(torch.float32), tspan, method='dopri5', rtol=1e-5, atol=1e-6)
    return sol

def loss_fn(pred, true):
    eps=1e-6
    weights = torch.tensor([0.2, 0.6, 0.2], device=DEVICE)
    scale = true.std(dim=0).clamp(min=eps)
    return torch.mean(weights * ((pred - true)/scale)**2)

# ---------------------------------------------------------------------------
# Entrenamiento: Fase 1 (Adam), Fase 2 (L-BFGS)
# ---------------------------------------------------------------------------
adam_iters = 1500
bfgs_iters = 200
loss_history = []
best_loss = float('inf')
best_state=None

print(f"Fase 1: Adam ({adam_iters} iters)")

for restart in range(5):
    torch.manual_seed(15*restart)
    nn_model=InfectionNN(inp_mean, inp_std).to(DEVICE)
    
    optimizer = optim.Adam([
        {'params': nn_model.parameters(), 'lr': 1e-2},
        {'params': [log_tau, log_omega], 'lr': 2e-3} 
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adam_iters, eta_min=1e-5)
    patience = 0
    restart_best=float('inf')

    for epoch in range(adam_iters):
        optimizer.zero_grad()
        pred = predict()
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        if loss.item() < restart_best:
            restart_best = loss.item()
            patience=0
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = copy.deepcopy({'nn': nn_model.state_dict(),
                                            'log_tau': log_tau.detach().clone(),
                                            'log_omega': log_omega.detach().clone()})
        else:
            patience += 1
            if patience >= 100:
                print(f"  Se alcanzó la paciencia máxima en restart {restart}\n")
                break
        if epoch % 50 == 0:
            print(f"  [SGD] iter {epoch} — Loss: {loss.item():.6f}")

print(f"  Loss final SGD: {loss_history[-1]:.6f}\n")


# Cargar solo los parámetros de la red neuronal
nn_model.load_state_dict(best_state['nn'])

# Actualizar los parámetros de la ODE fuera del estado de la red
log_tau.data = best_state['log_tau'].data
log_omega.data = best_state['log_omega'].data

print(f"Mejor modelo cargado con loss {best_loss:.6f} después de 5 reinicios\n")

print(f"Fase 2: L-BFGS de PyTorch\n")

def l2_reg(model, lambda_reg=1e-4):
    return lambda_reg*sum(p.pow(2).sum() for p in model.parameters())

# Usamos el LBFGS nativo de PyTorch
lbfgs_optimizer = optim.LBFGS(list(nn_model.parameters()) + [log_tau, log_omega], 
                              max_iter=bfgs_iters, 
                              history_size=10, 
                              tolerance_grad=1e-7, 
                              tolerance_change=1e-9, 
                              line_search_fn="strong_wolfe")

def closure():
    lbfgs_optimizer.zero_grad()
    pred = predict()
    loss = loss_fn(pred, target) + l2_reg(nn_model)
    loss.backward()
    return loss

lbfgs_optimizer.step(closure)

# Para registrar el loss final
pred_final = predict()
loss_final = loss_fn(pred_final, target)
loss_history.append(loss_final.item())
print(f"  Loss final L-BFGS: {loss_final.item():.6f}\n")
print(f"Entrenamiento finalizado. Loss final: {loss_history[-1]:.6f}")

# ---------------------------------------------------------------------------
# Evaluación del modelo entrenado
# ---------------------------------------------------------------------------
with torch.no_grad():
    sol_trained = odeint(sir_ude, u0.to(torch.float32), torch.linspace(0., MAX_DAYS_TOT, steps=MAX_DAYS_TOT, dtype=torch.float32),
                     method='dopri5', rtol=1e-5, atol=1e-6)
    loss_train=loss_fn(sol_trained[:MAX_DAYS_TRAIN], y_real_train)
    loss_test = loss_fn(sol_trained[MAX_DAYS_TRAIN:], y_real_full[MAX_DAYS_TRAIN:])

    print(f"Loss train: {loss_train.item():.6f}")
    print(f"Loss test: {loss_test.item():.6f}")

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
# Definimos el límite de validación
SPLIT_VAL = MAX_DAYS_TRAIN + MAX_DAYS_VAL

def plot_sirs_ude_results_separados(model, u0, t_span, sol_trained, I_obs_norm, S_obs_norm, R_obs_norm, casos_diarios_real, loss_hist, train_days, val_days, max_days, save_folder, N_pob):
    # Desacoplamos tensores
    S_ude = sol_trained[:, 0].detach().cpu().numpy()
    I_ude = sol_trained[:, 1].detach().cpu().numpy()
    R_ude = sol_trained[:, 2].detach().cpu().numpy()
    
    # Limpiamos la loss history por si quedaron tensores con gradientes
    loss_hist_limpio = [l.item() if torch.is_tensor(l) else l for l in loss_hist]
    
    # Calculamos Beta efectivo
    with torch.no_grad():
        estados = torch.tensor(np.stack([S_ude, I_ude, R_ude], axis=1), dtype=torch.float32).to(DEVICE)
        estados = torch.clamp(estados, 0.0, 1.0)
        beta_eff = model(estados).squeeze(-1).cpu().numpy()
    beta_eff = np.nan_to_num(beta_eff, nan=0.0, posinf=0.0, neginf=0.0)

    t_dias = np.arange(max_days)

    # Reconstrucción de incidencia diaria (Nuevos casos = beta * S * I * N)
    nuevos_casos_pred = beta_eff * S_ude * I_ude * N_pob
    
    millions_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')
    thousands_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e3:.0f}k')

    def aplicar_formato_base(ax, titulo, ylabel, xlabel="Días", y_formatter=None, mostrar_leyenda=True):
        # Líneas divisorias
        ax.axvline(x=train_days, color='#2c3e50', linestyle='--', lw=2.5, zorder=3, label='Fin Train')
        ax.axvline(x=val_days, color='#8e44ad', linestyle=':', lw=2.5, zorder=3, label='Fin Validación')
        
        # Sombreados para las 3 fases
        ax.axvspan(0, train_days, alpha=0.06, color='#3498db', label='Zona Train') 
        ax.axvspan(train_days, val_days, alpha=0.06, color='#f1c40f', label='Zona Validación')
        ax.axvspan(val_days, max_days, alpha=0.06, color='#e74c3c', label='Zona Test')
        
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
    plt.plot(loss_hist_limpio, color='#8e44ad', lw=2.5, label='Loss')
    plt.title("Evolución de la Función de Pérdida", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Loss (Escala Log)", fontsize=12)
    plt.xlabel("Iteraciones", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    path_loss = os.path.join(save_folder, "01_loss_evolution.png")
    plt.savefig(path_loss, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. GRÁFICO: Beta Efectivo Aprendido
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, beta_eff, color='#5A189A', lw=2.5, label=r"$\beta_{ef}(t)$ (Infección)")
    aplicar_formato_base(ax, r"Tasa de Infección Efectiva Aprendida ($\beta_{ef}$)", "Valor del parámetro")
    path_params = os.path.join(save_folder, "02_beta_efectivo.png")
    plt.savefig(path_params, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. GRÁFICO: Infectados Activos (Ajuste a Escala Poblacional)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.scatter(t_dias, I_obs_norm * N_pob, color='black', alpha=0.3, s=12, label="Datos Reales (I)")
    ax.plot(t_dias, I_ude * N_pob, color='#D55E00', lw=2.5, label="Predicción UDE (I)")
    
    # Calculamos métricas para poner en el gráfico
    mae_tr = np.mean(np.abs(I_ude[:train_days] - I_obs_norm[:train_days])) * N_pob
    mae_te = np.mean(np.abs(I_ude[val_days:] - I_obs_norm[val_days:])) * N_pob
    ax.text(0.02, 0.95, f"Train MAE: {mae_tr:,.0f}\nTest MAE: {mae_te:,.0f}", 
            transform=ax.transAxes, va="top", fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", alpha=0.9))
            
    aplicar_formato_base(ax, "Infectados Activos (Dinámica Latente)", "Personas", y_formatter=thousands_formatter)
    path_activos = os.path.join(save_folder, "03_active_infected.png")
    plt.savefig(path_activos, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. GRÁFICO: Dinámica de Compartimentos SIRS (Fracción Poblacional)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(t_dias, S_obs_norm, color="#0072B2", alpha=0.3, linestyle='-', label="S Reales")
    ax.plot(t_dias, S_ude, color="#0072B2", lw=2.5, label="S Predichos")
    ax.plot(t_dias, R_obs_norm, color="#009E73", alpha=0.3, linestyle='-', label="R Reales")
    ax.plot(t_dias, R_ude, color="#009E73", lw=2.5, label="R Predichos")
    aplicar_formato_base(ax, "Dinámica SIRS: Pérdida de Inmunidad", "Proporción de la Población")
    path_sirs = os.path.join(save_folder, "04_sirs_compartments.png")
    plt.savefig(path_sirs, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. GRÁFICO: Nuevos Casos Diarios (Incidencia)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.scatter(t_dias, casos_diarios_real[:max_days], color='black', alpha=0.3, s=15, label="Datos Reales Diarios")
    ax.plot(t_dias, nuevos_casos_pred, color='#e74c3c', lw=2.5, label="Predicción UDE (Nuevos/Día)")
    aplicar_formato_base(ax, "Incidencia: Nuevos Casos Diarios", "Infectados por Día", y_formatter=thousands_formatter)
    path_diarios = os.path.join(save_folder, "05_daily_cases.png")
    plt.savefig(path_diarios, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Todos los gráficos fueron exportados exitosamente en '{save_folder}':")
    print("   -> 01_loss_evolution.png")
    print("   -> 02_beta_efectivo.png")
    print("   -> 03_active_infected.png")
    print("   -> 04_sirs_compartments.png")
    print("   -> 05_daily_cases.png")

# Ejecución de la exportación por archivos separados
plot_sirs_ude_results_separados(
    model=nn_model, 
    u0=u0, 
    t_span=torch.linspace(0., MAX_DAYS_TOT, steps=MAX_DAYS_TOT, dtype=torch.float32), 
    sol_trained=sol_trained, 
    I_obs_norm=I_obs_norm_full, 
    S_obs_norm=S_obs_norm_full, 
    R_obs_norm=R_obs_norm_full, 
    casos_diarios_real=nuevos_casos_diarios, 
    loss_hist=loss_history, 
    train_days=MAX_DAYS_TRAIN, 
    val_days=SPLIT_VAL,
    max_days=MAX_DAYS_TOT,
    save_folder=save_folder,
    N_pob=N
)
plt.show()
