# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:23:29 2026

@author: david
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. Cargar los datos de ganadores para los 4 escenarios
# Sustituye estos nombres por los de tus archivos reales
# ==========================================
df_A = pd.read_csv('datos_ganadores_50runs_alterado.csv')[['n1', 'n2', 'n3', 'n4']]
df_B = pd.read_csv('datos_ganadores_50runs_alteradoplus.csv')[['n1', 'n2', 'n3', 'n4']]
df_C = pd.read_csv('datos_ganadores_50runs_alteradisimo.csv')[['n1', 'n2', 'n3', 'n4']]
df_D = pd.read_csv('datos_ganadores_50runs_alteradisimoplus.csv')[['n1', 'n2', 'n3', 'n4']]
# Calcular las 4 matrices de correlación
corr_A = df_A.corr()
corr_B = df_B.corr()
corr_C = df_C.corr()
corr_D = df_D.corr()

# ==========================================
# 2. Configurar la figura global (Cuadrícula 2x2)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
#fig.suptitle('Evolución de Sinergias Estratégicas en Agentes Ganadores', fontsize=18, fontweight='bold', y=0.95)

# Parámetros comunes para todos los heatmaps (¡SIN el parámetro cbar!)
heatmap_kws = {
    'annot': True,
    'cmap': 'coolwarm',
    'vmin': -0.6,
    'vmax': 0,
    'linewidths': 1,
    'linecolor': 'black',
    'annot_kws': {"size": 11}
}

# --- Escenario A (Arriba Izquierda) ---
# Aquí apagamos la barra (cbar=False)
sns.heatmap(corr_A, ax=axs[0, 0], cbar=False, **heatmap_kws)
axs[0, 0].set_title('A ($\\alpha=0.3, \\pi=0\%$)', fontsize=14, pad=10)

# --- Escenario B (Arriba Derecha) ---
# Aquí encendemos la barra (cbar=True)
sns.heatmap(corr_B, ax=axs[0, 1], cbar=True, **heatmap_kws)
axs[0, 1].set_title('B ($\\alpha=0.3, \\pi=0.2\%$)', fontsize=14, pad=10)

# --- Escenario C (Abajo Izquierda) ---
# Aquí apagamos la barra (cbar=False)
sns.heatmap(corr_C, ax=axs[1, 0], cbar=False, **heatmap_kws)
axs[1, 0].set_title('C ($\\alpha=0.5, \\pi=0\%$)', fontsize=14, pad=10)

# --- Escenario D (Abajo Derecha) ---
# Aquí encendemos la barra (cbar=True)
sns.heatmap(corr_D, ax=axs[1, 1], cbar=True, **heatmap_kws)
axs[1, 1].set_title('D ($\\alpha=0.5, \\pi=0.2\%$)', fontsize=14, pad=10)

# ==========================================
# 3. Ajustes estéticos finales y guardado
# ==========================================
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Guardar la imagen en alta calidad para el TFG (formato PDF o PNG)
plt.savefig('matriz_correlaciones_4escenarios.pdf', bbox_inches='tight')
plt.show()