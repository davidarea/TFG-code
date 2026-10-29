# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 17:21:17 2026

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, laplace

# ==========================================
# 1. DEFINICIÓN DE CLASES
# ==========================================

def g_xi(x):
    y= ((2/3**0.75) * x)/(1.0 + (x / 2.0)**4)
    return y

class Agente:
    def __init__(self, indice, efectivo_inicial, acciones_iniciales, tau_inicial, gamma_optimo):
        self.id = indice 
        self.efectivo = efectivo_inicial 
        self.acciones = acciones_iniciales
        self.riqueza_total = 0.0
        self.n1, self.n2, self.n3 = 0.0, 0.0, 0.0
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
        self.T_SIMULACION = kwargs.get('T_sim', 5000)
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
        if len(precios_hist) < 3: return 0, 0, 0
        log_retornos = np.log(precios_hist[1:] / precios_hist[:-1])
        I1 = np.mean(log_retornos)
        I3 = np.mean(log_retornos**2)
        cambio_log_retornos = log_retornos[1:] - log_retornos[:-1]
        I2 = np.mean(cambio_log_retornos)
        return I1, I2, I3

    def _inicializar_agentes(self):
        lista_agentes = []; rands_n = np.random.rand(self.N, 3); sum_rands_n = np.sum(rands_n, axis=1, keepdims=True); rands_n_norm = rands_n / sum_rands_n
        rands_tau = np.random.uniform(0.1, 5.0, size=self.N); rands_gamma = np.random.uniform(0.5, 1.5, size=self.N) 
        for i in range(self.N):
            agente = Agente(indice=i, efectivo_inicial=self.B_INIT, acciones_iniciales=self.PHI_INIT, tau_inicial=rands_tau[i], gamma_optimo=rands_gamma[i])
            agente.n1, agente.n2, agente.n3 = rands_n_norm[i]
            agente.calcular_riqueza(self.S_ACTUAL)
            lista_agentes.append(agente)
        return lista_agentes
    
    def _crear_nuevo_agente_individual(self):
        # -------------------------------------------------------------
        # PEQUEÑO FIX: Entrar con riqueza media para no morir al instante

        riquezas = [a.riqueza_total for a in self.agentes]
        riqueza_media = np.mean(riquezas) if riquezas else 20000.0
        valor_acciones = self.PHI_INIT * self.S_ACTUAL
        nuevo_efectivo = max(100, riqueza_media - valor_acciones)
        # -------------------------------------------------------------
        
        self.MAX_ID += 1
        rands_n = np.random.rand(3); rands_n_norm = rands_n / np.sum(rands_n)
        rands_tau = np.random.uniform(0.1, 5.0); rands_gamma = np.random.uniform(0.5, 1.5)
        
        nuevo_agente = Agente(self.MAX_ID, nuevo_efectivo, self.PHI_INIT, rands_tau, rands_gamma)
        nuevo_agente.n1, nuevo_agente.n2, nuevo_agente.n3 = rands_n_norm
        nuevo_agente.calcular_riqueza(self.S_ACTUAL) 
        return nuevo_agente

    def calcular_expectativa_agentes(self, I1, I2, I3): 
        I_k = np.array([I1, I2, I3])
        log_S_actual = np.log(self.S_ACTUAL)
        for agente in self.agentes:
            n_ik = np.array([agente.n1, agente.n2, agente.n3])
            agente.x_i = np.dot(n_ik, I_k) 
            agente.g_xi = g_xi(agente.x_i)
            agente.S_esperado = np.exp(log_S_actual + agente.g_xi)

    def simular_mercado(self):
        trayectoria_S = list(self.historial_precios)
        
        # Para evitar spam en consola, quitamos el print del bucle o lo reducimos
        for t in range(self.T_SIMULACION):
            historial_relevante = np.array(trayectoria_S[-(self.T_PASADO + 1):])
            I1, I2, I3 = self.calcular_indicadores_T(historial_relevante)
            self.S_ACTUAL = trayectoria_S[-1]
            self.calcular_expectativa_agentes(I1, I2, I3)
            
            delta_phis_deseadas = []; D_t = 0.0; O_t = 0.0 
            for agente in self.agentes:
                delta_phi_i_deseada = agente.calcular_delta_phi(self.S_ACTUAL)
                delta_phis_deseadas.append(delta_phi_i_deseada)
                if delta_phi_i_deseada > 0: D_t += delta_phi_i_deseada
                elif delta_phi_i_deseada < 0: O_t += -delta_phi_i_deseada
            
            delta_phis_reales = []
            if D_t > O_t and D_t > 0: 
                factor = O_t / D_t
                for dp in delta_phis_deseadas: delta_phis_reales.append(dp * factor if dp > 0 else dp)
            elif O_t > D_t and O_t > 0: 
                factor = D_t / O_t
                for dp in delta_phis_deseadas: delta_phis_reales.append(dp if dp > 0 else dp * factor)
            else: delta_phis_reales = delta_phis_deseadas
                
            S_t = self.S_ACTUAL
            promedio_D = np.mean(self.historial_D)
            promedio_O = np.mean(self.historial_O)
            cociente_DO = 1.0 
            if promedio_D > 0 and promedio_O > 0: cociente_DO = promedio_D / promedio_O
            elif promedio_D > 0: cociente_DO = 1.0 + self.MAX_AJUSTE_PERCENTUAL 
            elif promedio_O > 0: cociente_DO = 1.0 - self.MAX_AJUSTE_PERCENTUAL 
            
            factor_cambio_atenuado = 1.0 + self.FACTOR_ATENUACION_ALFA * (cociente_DO - 1.0)
            S_tendencia = S_t * factor_cambio_atenuado
            ruido = np.random.normal(0, self.SIGMA * S_t)
            S_t_plus_1 = max(S_tendencia + ruido, 0.01) 
            trayectoria_S.append(S_t_plus_1)
            
            S_ejecucion = S_t_plus_1; e_t = self.TASA_INGRESO_E; pi = self.COSTO_TRANSACCION_PI
            for i, agente in enumerate(self.agentes):
                agente.actualizar_cartera_financiera(delta_phis_reales[i], S_ejecucion, e_t, pi)
                agente.calcular_riqueza(S_t_plus_1)
                
            # Evolución
            if (t + 1) % 10 == 0:
                riquezas = np.array([a.riqueza_total for a in self.agentes])
                indice_a_eliminar = np.argmin(riquezas)
                self.agentes.pop(indice_a_eliminar)
                self.agentes.append(self._crear_nuevo_agente_individual())
                
        return np.array(trayectoria_S)

# ==========================================
# 2. BUCLE DE 100 EJECUCIONES Y ANÁLISIS
# ==========================================

PARAMETROS = {
    'N_agentes': 1000, 'S_init': 100.0, 'B_init': 5000.0, 'phi_init': 50.0,
    'T_pasado': 5, 'S_min': 90.0, 'S_max': 110.0, 'T_sim': 5000, 'sigma_ruido': 0.02,
    'factor_atenuacion_alfa': 0.1, 'max_ajuste_percentual': 0.05,
    'tasa_ingreso_e': 0.0, 'costo_transaccion_pi': 0.0 
}

NUM_RUNS = 100
resultados_precios = {} # Guardar columnas

print(f"🚀 Iniciando batería de {NUM_RUNS} simulaciones...")

for i in range(NUM_RUNS):
    print(f"   ▶ Ejecutando Simulación {i+1}/{NUM_RUNS}...")
    simulacion = Mercado(**PARAMETROS)
    # Ejecutamos y guardamos solo la parte simulada 
    trayectoria_completa = simulacion.simular_mercado()
    resultados_precios[f'Run_{i+1}'] = trayectoria_completa

# Crear DataFrame con los precios
df_precios = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resultados_precios.items() ]))
df_precios.to_csv('datos_simulacion_100_runs.csv', index=False)
print("✅ Datos de precios guardados en 'datos_simulacion_100_runs.csv'")

# Volvemos a importar esto porque al tener ya los datos simulados podemos sólo correr esta parte y obtener nuestros resultados

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, laplace

df_precios = pd.read_csv('datos_precios_simulacion_20runs_alterado.csv')

# ==========================================
# 3. ANÁLISIS 
# ==========================================

print("Calculando Retornos Logarítmicos Normalizados...")

# 1. Calcular LOG-RETORNOS (r_t = ln(S_t) - ln(S_t-1))
# Esto elimina el efecto de si el precio es 100 o 200, mide porcentajes.
log_precios = np.log(df_precios)
log_retornos = log_precios.diff().dropna().values.flatten()

# 2. NORMALIZAR
# Restamos la media y dividimos por la desviación estándar.
# Esto centra la gráfica en 0 y hace que la desviación estándar sea 1.
# Así podemos compararla con una Normal Estándar N(0,1).
mu_datos = np.mean(log_retornos)
sigma_datos = np.std(log_retornos)
datos_normalizados = (log_retornos - mu_datos) / sigma_datos

# ==========================================
# 4. GRAFICAR
# ==========================================
plt.figure(figsize=(10, 7))

# A. Histograma de mis datos (Azul)
plt.hist(datos_normalizados, bins=150, density=True, 
         alpha=0.5, color='blue', label='Retornos Normalizados')

# B. Línea de Referencia 1: GAUSSIANA (Roja)
# La teoría clásica dice que debería seguir esto.
x = np.linspace(-6, 6, 1000)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=3, label='Distribución Normal')

# C. Línea de Referencia 2: LAPLACE (Verde)
# Esta es la forma de "Tienda de Campaña".
# Suele ajustar mejor a los mercados reales que la Gaussiana.
plt.plot(x, laplace.pdf(x, 0, 1/np.sqrt(2)), 'g--', linewidth=2, label='Distribución de Laplace')

# D. AJUSTES VISUALES CRÍTICOS
plt.yscale('log')  # Escala Logarítmica


#Cortamos eje y
plt.ylim(bottom=0.000001, top=5.0) 
plt.xlim(-8,8) # Nos centramos en ±6 desviaciones estándar (donde están los eventos raros)

plt.xlabel('Retorno Normalizado (Desviaciones Estándar)')
plt.ylabel('Densidad de Probabilidad (Log)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()







import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, laplace, cauchy # <-- Añadida cauchy aquí

df_precios = pd.read_csv('datos_precios_simulacion_20runs_alterado.csv')

# ==========================================
# 3. ANÁLISIS 
# ==========================================

print("Calculando Retornos Logarítmicos Normalizados...")

# 1. Calcular LOG-RETORNOS (r_t = ln(S_t) - ln(S_t-1))
# Esto elimina el efecto de si el precio es 100 o 200, mide porcentajes.
log_precios = np.log(df_precios)
log_retornos = log_precios.diff().dropna().values.flatten()

# 2. NORMALIZAR
# Restamos la media y dividimos por la desviación estándar.
# Esto centra la gráfica en 0 y hace que la desviación estándar sea 1.
# Así podemos compararla con una Normal Estándar N(0,1).
mu_datos = np.mean(log_retornos)
sigma_datos = np.std(log_retornos)
datos_normalizados = (log_retornos - mu_datos) / sigma_datos

# ==========================================
# 4. GRAFICAR
# ==========================================
plt.figure(figsize=(10, 7))

# A. Histograma de mis datos (Azul)
plt.hist(datos_normalizados, bins=150, density=True, 
         alpha=0.5, color='blue', label='Retornos Normalizados')

# B. Línea de Referencia 1: GAUSSIANA (Roja)
# La teoría clásica dice que debería seguir esto.
x = np.linspace(-8, 8, 1000) # Ampliado a 8 para cubrir todo el eje X
plt.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=3, label='Distribución Normal')

# C. Línea de Referencia 2: LAPLACE (Verde)
# Esta es la forma de "Tienda de Campaña".
# Suele ajustar mejor a los mercados reales que la Gaussiana.
plt.plot(x, laplace.pdf(x, 0, 1/np.sqrt(2)), 'g--', linewidth=2, label='Distribución de Laplace')

# D. Línea de Referencia 3: LORENTZIANA / CAUCHY (Morada)
# Famosa en física como Lorentziana, en estadística es la Cauchy.
# Tiene colas extremadamente pesadas (varianza indefinida matemáticamente).
plt.plot(x, cauchy.pdf(x, 0, 1), color='purple', linestyle='-', linewidth=3, label='Distribución Lorentziana')

# E. AJUSTES VISUALES CRÍTICOS
plt.yscale('log')  # Escala Logarítmica

# Cortamos eje y y x para centrar la vista en lo importante
plt.ylim(bottom=0.000001, top=5.0) 
plt.xlim(-8, 8) 

# Etiquetas y Leyenda
plt.xlabel('Retorno Normalizado (Desviaciones Estándar)', fontsize=14)
plt.ylabel('Densidad de Probabilidad (Log)', fontsize=14)

# Quitado el grid como me pediste en imágenes anteriores y ajustada leyenda
plt.legend(fontsize=12)
plt.grid(False)

plt.tight_layout()
plt.show()






