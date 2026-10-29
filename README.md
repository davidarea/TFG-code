# Agent-Based Financial Market Simulation

## Overview
This project implements a microscopic agent-based model (ABM) of a financial market from an Econophysics perspective. Based on Caldarelli's formalism, the model simulates how millions of heterogeneous investors interact through a simplified order book, generating complex price dynamics that mimic real-world financial markets.

Unlike standard financial models (such as Black-Scholes or Geometric Brownian Motion) that assume normal distributions and constant volatility, this simulation successfully reproduces empirical stylized facts of real markets endogenously. Furthermore, the model incorporates an evolutionary dynamic: every 10 temporal steps, the poorest-performing agent is eliminated and replaced by a mutated version of a top-performing investor, simulating natural selection in trading strategies.

## Key Scientific Features
- **Emergent Market Phenomena:** The simulation successfully reproduces heavy tails in return distributions and volatility clustering without hardcoding them.
- **Heterogeneous Strategies:** Agents form expectations using a weighted combination of four market indicators:
  1. Trend ($I_1$)
  2. Acceleration / Momentum ($I_2$)
  3. Historical Volatility ($I_3$)
  4. Contrarian / Mean-reversion ($I_4$) — *Original contribution to the model*
- **Evolutionary Selection:** Analysis of how different strategies (e.g., trend following vs. mean-reversion) thrive or collapse under varying market reactivity ($\alpha$) and transaction costs ($\pi$).

## Objectives
- Simulate a financial market populated by interacting, heterogeneous investors.
- Evaluate the impact of cognitive biases and trading preferences on asset price formation.
- Prove that stochastic diffusion and collective microscopic behaviors are the common threads connecting statistical mechanics and finance.
- Analyze the evolutionary correlation and survival of specific trading strategies under different economic frictions.

## Repository Structure
The code used for the simulation and data analysis is organized as follows:

- `src/main_simulacion.py`: The core engine of the ABM. Contains the `Agente` and `Mercado` classes and runs the evolutionary simulation, exporting the price and strategy data to CSV.
- `src/plot_colas_pesadas.py`: Analyzes the generated logarithmic returns and plots them against theoretical distributions (Gaussian, Laplace, Cauchy) to demonstrate the emergence of heavy tails.
- `src/plot_autocorrelacion.py`: Calculates and plots the temporal autocorrelation of returns and absolute returns to visualize volatility clustering.
- `src/plot_correlaciones.py`: Generates heatmaps of the strategy weights (indicators $I_1$ to $I_4$) of the winning agents across 4 different simulation scenarios (varying reactivity and transaction costs).

## Technologies
- **Python 3**
- Data Analysis & Math: `numpy`, `pandas`, `scipy`
- Visualization: `matplotlib`, `seaborn`

## Project Status
Completed as a Bachelor's Thesis (Trabajo de Fin de Grado) in Physics at Universidade de Santiago de Compostela (USC), Academic Year 2025-2026.
