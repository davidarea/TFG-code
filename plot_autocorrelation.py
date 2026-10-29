# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:29:49 2026

@author: david
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar tus datos (Asegúrate de que el nombre del CSV sea el correcto)
df_precios = pd.read_csv('datos_precios_simulacion_50runs_alterado.csv')

MAX_LAGS = 15
autocorr_retornos = []
autocorr_volatilidad = []

# 2. Calcular la Autocorrelación para cada una de las simulaciones
for columna in df_precios.columns:
    precios = df_precios[columna].values
    
    # Retornos logarítmicos normales (r_t)
    retornos = np.log(precios[1:] / precios[:-1])
    serie_retornos = pd.Series(retornos)
    
    # Retornos absolutos (|r_t|) - Esto mide la Volatilidad
    serie_volatilidad = pd.Series(np.abs(retornos))
    
    # Calcular C(tau) para cada lag
    acf_ret = [serie_retornos.autocorr(lag=i) for i in range(MAX_LAGS)]
    acf_vol = [serie_volatilidad.autocorr(lag=i) for i in range(MAX_LAGS)]
    
    autocorr_retornos.append(acf_ret)
    autocorr_volatilidad.append(acf_vol)

# 3. Hacer la media de las runs (para alisar la curva)
media_acf_ret = np.mean(np.array(autocorr_retornos), axis=0)
media_acf_vol = np.mean(np.array(autocorr_volatilidad), axis=0)

# ==========================================
# 4. GRAFICAR LOS RESULTADOS
# ==========================================
plt.figure(figsize=(10,7))

# Líneas de referencia visual
plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
plt.axhline(0.05, color='gray', linestyle='--', alpha=0.5) # Bandas de "ruido estadístico"
plt.axhline(-0.05, color='gray', linestyle='--', alpha=0.5)

# Dibujamos la autocorrelación de los retornos (Sin memoria)
plt.plot(range(MAX_LAGS), media_acf_ret, color='royalblue', marker='o', linestyle='-', 
         markersize=5, label='Retornos Logarítmicos $r_t$')

# Dibujamos la autocorrelación de la volatilidad (Con memoria / Clustering)
plt.plot(range(MAX_LAGS), media_acf_vol, color='firebrick', marker='s', linestyle='-', 
         markersize=5, label='Volatilidad $|r_t|$')

# Ajustes de la gráfica
plt.xlabel('Retardo $\\tau$ (pasos temporales)', fontsize=14)
plt.ylabel('Autocorrelación', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()