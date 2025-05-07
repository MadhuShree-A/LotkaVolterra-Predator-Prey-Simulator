# Lotka-Volterra-Predator-Prey-Simulator
Lotka-Volterra Predator-Prey Simulator

This is a Python application for simulating and analyzing the **Lotka-Volterra equations**, a pair of first-order, nonlinear, differential equations frequently used to describe the dynamics of biological systems such as predator-prey interactions.

It provides:
- Basic simulations of prey and predator population dynamics
- Bifurcation analysis by varying the prey birth rate (α)
- Stochastic modeling with noise
- Sensitivity analysis using Jacobian eigenvalues
- Time series analysis with autocorrelation and Fourier transform


##  Requirements

- Python 3.7+
- Libraries:
  ```bash
  pip install numpy matplotlib scipy seaborn

##  How to Run
git clone https://github.com/your-username/lotka-volterra-simulator.git
- cd lotka-volterra-simulator

## Run the main script:
python your_script_name.py

- A GUI window will open where you can enter parameters and run simulations.

## Parameters

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| α (alpha) | Prey birth rate                                     |
| β (beta)  | Predation rate                                      |
| δ (delta) | Predator reproduction rate                          |
| γ (gamma) | Predator death rate                                 |
| x₀        | Initial prey population                             |
| y₀        | Initial predator population                         |
| t_end     | Simulation time range                                |
| Noise     | Gaussian noise scale (only used in stochastic mode) |

