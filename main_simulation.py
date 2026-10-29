# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:09:59 2026

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Función g(x) ---
def g_xi(x):
    y= ((2/3**0.75) * x)/(1.0 + (x / 2.0)**4)
    return y

# --- 2. Clase Agente 
class Agente:
    def __init__(self, indice, efectivo_inicial, acciones_iniciales, tau_inicial, gamma_optimo):
        self.id = indice 
        self.efectivo = efectivo_inicial 
        self.acciones = acciones_iniciales
        self.riqueza_total = 0.0
        self.n1, self.n2, self.n3, self.n4 = 0.0, 0.0, 0.0, 0.0
        self.tau = tau_inicial
        self.gamma_opt = gamma_optimo
        self.x_i, self.g_xi, self.S_esperado = 0.0, 0.0, 0.0

    def calcular_riqueza(self, precio_stock):
        valor_acciones = self.acciones * precio_stock
        self.riqueza_total = self.efectivo + valor_acciones
        return self.riqueza_total
    
    def calcular_delta_phi(self, precio_stock):
        termino_expectativa = self.g_xi * self.acciones
        desajuste_efectivo = self.gamma_opt * self.efectivo - (self.acciones * precio_stock)
        termino_ajuste = desajuste_efectivo / 2.0 * self.tau
        delta_phi_bruta = termino_expectativa + termino_ajuste
        
        precio_seguro = max(precio_stock, 0.01)
        if delta_phi_bruta > 0: 
            max_compra = self.efectivo / precio_seguro
            delta_phi_deseada = min(delta_phi_bruta, max_compra)
        elif delta_phi_bruta < 0: 
            max_venta = -self.acciones 
            delta_phi_deseada = max(delta_phi_bruta, max_venta)
        else:
            delta_phi_deseada = 0.0
            
        return delta_phi_deseada

    def actualizar_cartera_financiera(self, delta_phi_real, precio_transaccion, tasa_ingreso_e, costo_transaccion_pi):
        costo_total = precio_transaccion * (1.0 + costo_transaccion_pi) * delta_phi_real
        self.acciones += delta_phi_real
        ingreso_efectivo = (1.0 + tasa_ingreso_e) * self.efectivo
        self.efectivo = ingreso_efectivo - costo_total
        self.efectivo = max(self.efectivo, 0.0)
        self.acciones = max(self.acciones, 0.0)

# --- 3. Clase Mercado/Simulación ---
class Mercado:
    def __init__(self, **kwargs):
        self.N = kwargs.get('N_agentes', 1000)
        self.S_INIT = kwargs.get('S_init', 100.0)
        self.B_INIT = kwargs.get('B_init', 20000.0)
        self.PHI_INIT = kwargs.get('phi_init', 50.0)
        self.T_PASADO = kwargs.get('T_pasado', 10)
        self.S_MIN = kwargs.get('S_min', 90.0)
        self.S_MAX = kwargs.get('S_max', 110.0)
        self.SIGMA = kwargs.get('sigma_ruido', 0.02)
        self.MAX_ID = self.N
        self.T_SIMULACION = kwargs.get('T_sim', 25000)
        self.TASA_INGRESO_E = kwargs.get('tasa_ingreso_e', 0.0) 
        self.COSTO_TRANSACCION_PI = kwargs.get('costo_transaccion_pi', 0.0)
        self.FACTOR_ATENUACION_ALFA = kwargs.get('factor_atenuacion_alfa', 0.05) 
        self.MAX_AJUSTE_PERCENTUAL = kwargs.get('max_ajuste_percentual', 0.05) 
        
        self.historial_precios = self._generar_historial_precios()
        self.S_ACTUAL = self.historial_precios[-1]
        self.agentes = self._inicializar_agentes()
        
        self.historial_D = np.zeros(self.T_PASADO) 
        self.historial_O = np.zeros(self.T_PASADO) 
        
    def _generar_historial_precios(self):
        historial_precios = np.random.uniform(self.S_MIN, self.S_MAX, size=self.T_PASADO)
        return np.append(historial_precios, self.S_INIT)

    def calcular_indicadores_T(self, precios_hist):
        if len(precios_hist) < 3: return 0, 0, 0, 0
        log_retornos = np.log(precios_hist[1:] / precios_hist[:-1])
        I1 = np.mean(log_retornos)
        I3 = np.mean(log_retornos**2)
        cambio_log_retornos = log_retornos[1:] - log_retornos[:-1]
        I2 = np.mean(cambio_log_retornos)
        I4 = -log_retornos[-1]
        
        return I1, I2, I3, I4

    def _inicializar_agentes(self):
        lista_agentes = []
        rands_n = np.random.rand(self.N, 4) 
        sum_rands_n = np.sum(rands_n, axis=1, keepdims=True)
        rands_n_norm = rands_n / sum_rands_n
        
        rands_tau = np.random.uniform(0.1, 5.0, size=self.N)
        rands_gamma = np.random.uniform(0.5, 1.5, size=self.N) 
        
        for i in range(self.N):
            agente = Agente(indice=i, efectivo_inicial=self.B_INIT, acciones_iniciales=self.PHI_INIT, tau_inicial=rands_tau[i], gamma_optimo=rands_gamma[i])
            agente.n1, agente.n2, agente.n3, agente.n4 = rands_n_norm[i]
            agente.calcular_riqueza(self.S_ACTUAL)
            lista_agentes.append(agente)
        return lista_agentes
    
    def _crear_nuevo_agente_individual(self):
        self.MAX_ID += 1
        riquezas = np.array([a.riqueza_total for a in self.agentes])
        riqueza_entrada = np.percentile(riquezas, 50) 
        valor_acciones = self.PHI_INIT * self.S_ACTUAL
        nuevo_efectivo = max(100.0, riqueza_entrada - valor_acciones)
        
        umbral_exito = np.percentile(riquezas, 80)
        mentores_potenciales = [a for a in self.agentes if a.riqueza_total >= umbral_exito]
        
        riquezas_mentores = np.array([a.riqueza_total for a in mentores_potenciales])
        probabilidades = riquezas_mentores / np.sum(riquezas_mentores)
        mentor = np.random.choice(mentores_potenciales, p=probabilidades)
        
        tasa_mutacion = 0.05 
        nuevos_n = np.array([mentor.n1, mentor.n2, mentor.n3, mentor.n4]) + np.random.normal(0, tasa_mutacion, 4)
        nuevos_n = np.clip(nuevos_n, 0.001, None) 
        rands_n_norm = nuevos_n / np.sum(nuevos_n) 
        
        nuevo_tau = max(0.1, mentor.tau + np.random.normal(0, 0.2))
        nuevo_gamma = np.clip(mentor.gamma_opt + np.random.normal(0, 0.05), 0.1, 2.0)
        
        nuevo_agente = Agente(self.MAX_ID, nuevo_efectivo, self.PHI_INIT, nuevo_tau, nuevo_gamma)
        nuevo_agente.n1, nuevo_agente.n2, nuevo_agente.n3, nuevo_agente.n4 = rands_n_norm
        nuevo_agente.calcular_riqueza(self.S_ACTUAL) 
        
        return nuevo_agente

    def calcular_expectativa_agentes(self, I1, I2, I3, I4): 
        I_k = np.array([I1, I2, I3, I4])
        log_S_actual = np.log(self.S_ACTUAL)
        for agente in self.agentes:
            n_ik = np.array([agente.n1, agente.n2, agente.n3, agente.n4])
            agente.x_i = np.dot(n_ik, I_k) 
            agente.g_xi = g_xi(agente.x_i)
            agente.S_esperado = np.exp(log_S_actual + agente.g_xi)

    def simular_mercado(self):
        trayectoria_S = list(self.historial_precios)
        
        for t in range(self.T_SIMULACION):
            historial_relevante = np.array(trayectoria_S[-(self.T_PASADO + 1):])
            I1, I2, I3, I4 = self.calcular_indicadores_T(historial_relevante) # MODIFICADO
            self.S_ACTUAL = trayectoria_S[-1]
            self.calcular_expectativa_agentes(I1, I2, I3, I4) # MODIFICADO
            
            delta_phis_deseadas = []; D_t = 0.0; O_t = 0.0 
            for agente in self.agentes:
                delta_phi_i_deseada = agente.calcular_delta_phi(self.S_ACTUAL) 
                delta_phis_deseadas.append(delta_phi_i_deseada)

                if delta_phi_i_deseada > 0: D_t += delta_phi_i_deseada
                elif delta_phi_i_deseada < 0: O_t += -delta_phi_i_deseada
            
            delta_phis_reales = []; factor_ajuste = 1.0
            
            if D_t > O_t and D_t > 0: 
                factor_ajuste = O_t / D_t
                for delta_phi in delta_phis_deseadas:
                    if delta_phi > 0: delta_phis_reales.append(delta_phi * factor_ajuste)
                    else: delta_phis_reales.append(delta_phi)
            elif O_t > D_t and O_t > 0: 
                factor_ajuste = D_t / O_t
                for delta_phi in delta_phis_deseadas:
                    if delta_phi > 0: delta_phis_reales.append(delta_phi)
                    else: delta_phis_reales.append(delta_phi * factor_ajuste)
            else: 
                delta_phis_reales = delta_phis_deseadas
                
            S_t = self.S_ACTUAL
            promedio_D = np.mean(self.historial_D)
            promedio_O = np.mean(self.historial_O)
            
            cociente_DO = 1.0 
            if promedio_D > 0 and promedio_O > 0: cociente_DO = promedio_D / promedio_O
            elif promedio_D > 0 and promedio_O == 0: cociente_DO = 1.0 + self.MAX_AJUSTE_PERCENTUAL 
            elif promedio_D == 0 and promedio_O > 0: cociente_DO = 1.0 - self.MAX_AJUSTE_PERCENTUAL 
            
            factor_cambio_atenuado = 1.0 + self.FACTOR_ATENUACION_ALFA * (cociente_DO - 1.0)
            S_tendencia = S_t * factor_cambio_atenuado
            ruido = np.random.normal(0, self.SIGMA * S_t)
            S_t_plus_1 = max(S_tendencia + ruido, 0.01) 
            trayectoria_S.append(S_t_plus_1)
            
            S_ejecucion = S_t_plus_1; e_t = self.TASA_INGRESO_E; pi = self.COSTO_TRANSACCION_PI

            for i, agente in enumerate(self.agentes):
                delta_phi_real = delta_phis_reales[i]
                agente.actualizar_cartera_financiera(delta_phi_real, S_ejecucion, e_t, pi)
                agente.calcular_riqueza(S_t_plus_1)
                
            if (t + 1) % 10 == 0:
                riquezas = np.array([a.riqueza_total for a in self.agentes])
                indice_a_eliminar = np.argmin(riquezas); agente_eliminado = self.agentes.pop(indice_a_eliminar)
                nuevo_agente = self._crear_nuevo_agente_individual()
                self.agentes.append(nuevo_agente)
                
        return np.array(trayectoria_S)

# ==========================================
# BUCLE DE EXTRACCIÓN Y GUARDADO
# ==========================================

PARAMETROS = {
    'N_agentes': 1000, 'S_init': 100.0, 'B_init': 5000.0, 'phi_init': 50.0,
    'T_pasado': 5, 'S_min': 90.0, 'S_max': 110.0, 'T_sim': 25000, 'sigma_ruido': 0.005,
    'factor_atenuacion_alfa': 0.1, 'max_ajuste_percentual': 0.05,
    'tasa_ingreso_e': 0.0, 'costo_transaccion_pi': 0.0 
}

NUM_RUNS = 20

n1_todos, n2_todos, n3_todos, n4_todos = [], [], [], []
n1_ganadores, n2_ganadores, n3_ganadores, n4_ganadores = [], [], [], []
resultados_precios = {}

print(f"🚀 Iniciando {NUM_RUNS} simulaciones para analizar estrategias...")

for i in range(NUM_RUNS):
    print(f"   ▶ Ejecutando Simulación {i+1}/{NUM_RUNS}...")
    simulacion = Mercado(**PARAMETROS)
    trayectoria_S = simulacion.simular_mercado()
    resultados_precios[f'Run_{i+1}'] = trayectoria_S
    
    agentes_finales = simulacion.agentes
    
    for a in agentes_finales:
        n1_todos.append(a.n1)
        n2_todos.append(a.n2)
        n3_todos.append(a.n3)
        n4_todos.append(a.n4) 
        
    ganador = max(agentes_finales, key=lambda x: x.riqueza_total)
    n1_ganadores.append(ganador.n1)
    n2_ganadores.append(ganador.n2)
    n3_ganadores.append(ganador.n3)
    n4_ganadores.append(ganador.n4)

print("✅ Datos extraídos correctamente. Guardando en CSV...")

# --- GUARDAR DATOS EN CSV ---
df_poblacion = pd.DataFrame({'n1': n1_todos, 'n2': n2_todos, 'n3': n3_todos, 'n4': n4_todos})
df_poblacion.to_csv('datos_poblacion_general_20runs_pocoruido.csv', index=False)

df_ganadores = pd.DataFrame({'n1': n1_ganadores, 'n2': n2_ganadores, 'n3': n3_ganadores, 'n4': n4_ganadores})
df_ganadores.to_csv('datos_ganadores_20runs_pocoruido.csv', index=False)

df_precios = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in resultados_precios.items() ]))
df_precios.to_csv('datos_precios_simulacion_20runs_pocoruido.csv', index=False)

print("💾 ¡Todo guardado con éxito! (Precios y 4 Estrategias)")
