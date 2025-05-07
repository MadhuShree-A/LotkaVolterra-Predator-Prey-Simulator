import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
import seaborn as sns
from scipy.signal import correlate
from scipy.fft import fft, fftfreq

def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z 
    dxdt = alpha * x - beta * x * y  
    dydt = delta * x * y - gamma * y  
    return [dxdt, dydt]

def simulate_and_plot(alpha, beta, delta, gamma, initial_conditions, t_span, t_eval, plot_phase_plane=False):
    solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, delta, gamma), t_eval=t_eval, method='RK45')
    
    prey_population = solution.y[0]
    predator_population = solution.y[1]
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, prey_population, label='Prey (x)', color='blue')
    plt.plot(t_eval, predator_population, label='Predator (y)', color='red')
    plt.fill_between(t_eval, prey_population, color='blue', alpha=0.1)
    plt.fill_between(t_eval, predator_population, color='red', alpha=0.1)
    plt.title(f'Time Series of Prey and Predator Populations\n(α={alpha}, β={beta}, δ={delta}, γ={gamma})')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plot_histograms(prey_population, predator_population)

    if plot_phase_plane:
        plt.figure(figsize=(6, 6))
        plt.hexbin(prey_population, predator_population, gridsize=30, cmap='Blues', reduce_C_function=np.mean)  # Corrected line
        plt.colorbar(label='Density')
        plt.title('Phase Plane: Predator vs. Prey Population')
        plt.xlabel('Prey Population (x)')
        plt.ylabel('Predator Population (y)')
        plt.grid(True)
        plt.show()

def plot_histograms(prey_population, predator_population):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(prey_population, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Prey Population')
    plt.xlabel('Population Size')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(predator_population, bins=30, color='red', alpha=0.7)
    plt.title('Histogram of Predator Population')
    plt.xlabel('Population Size')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def bifurcation_analysis(initial_conditions, t_span, t_eval):
    alpha_values = np.linspace(0.05, 0.2, 10)  
    plt.figure(figsize=(10, 6))
    
    for alpha in alpha_values:
        solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, 0.02, 0.01, 0.1), t_eval=t_eval, method='RK45')
        plt.plot(t_eval, solution.y[0], label=f'α={alpha:.2f}')  # Plot prey population
        
    plt.title('Bifurcation Analysis: Varying Prey Birth Rate (α)')
    plt.xlabel('Time')
    plt.ylabel('Prey Population')
    plt.legend()
    plt.grid(True)
    plt.show()

def lotka_volterra_stochastic(t, z, alpha, beta, delta, gamma, noise_scale):
    x, y = z
    dxdt = alpha * x - beta * x * y + np.random.normal(0, noise_scale) 
    dydt = delta * x * y - gamma * y + np.random.normal(0, noise_scale)  
    return [dxdt, dydt]

def simulate_stochastic(alpha, beta, delta, gamma, initial_conditions, t_span, t_eval, noise_scale):
    solution = solve_ivp(lotka_volterra_stochastic, t_span, initial_conditions, args=(alpha, beta, delta, gamma, noise_scale), t_eval=t_eval, method='RK45')
    
    prey_population = solution.y[0]
    predator_population = solution.y[1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, prey_population, label='Prey (x)', color='blue')
    plt.plot(t_eval, predator_population, label='Predator (y)', color='red')
    plt.title(f'Stochastic Lotka-Volterra Model with Noise Scale {noise_scale}')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()

def jacobian_matrix(alpha, beta, delta, gamma, x_eq, y_eq):
    J = np.array([[alpha - beta * y_eq, -beta * x_eq],
                  [delta * y_eq, delta * x_eq - gamma]])
    return J

def sensitivity_analysis(alpha, beta, delta, gamma):
    x_eq = gamma / delta
    y_eq = alpha / beta
    
    J = jacobian_matrix(alpha, beta, delta, gamma, x_eq, y_eq)
    
    eigenvalues, _ = np.linalg.eig(J)
    interpret_sensitivity(eigenvalues)
    
    print(f"Jacobian matrix at equilibrium (x={x_eq:.2f}, y={y_eq:.2f}):")
    print(J)
    print(f"Eigenvalues of the Jacobian: {eigenvalues}")
    
def analyze_time_series(prey_population, predator_population, t_eval):
    prey_autocorr = correlate(prey_population - np.mean(prey_population), prey_population - np.mean(prey_population), mode='full') / np.var(prey_population)
    predator_autocorr = correlate(predator_population - np.mean(predator_population), predator_population - np.mean(predator_population), mode='full') / np.var(predator_population)
    
    time_shifts = np.linspace(-len(t_eval), len(t_eval), 2*len(t_eval)-1)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_shifts, prey_autocorr, label='Prey Autocorrelation', color='blue')
    plt.title('Autocorrelation of Prey Population')
    plt.xlabel('Time Shift')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_shifts, predator_autocorr, label='Predator Autocorrelation', color='red')
    plt.title('Autocorrelation of Predator Population')
    plt.xlabel('Time Shift')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    prey_fft = fft(prey_population)
    predator_fft = fft(predator_population)
    
    freqs = fftfreq(len(t_eval), t_eval[1] - t_eval[0])
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(prey_fft), color='blue')
    plt.title('Fourier Transform of Prey Population')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(predator_fft), color='red')
    plt.title('Fourier Transform of Predator Population')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_simulation():
    try:
        alpha = float(entry_alpha.get())
        beta = float(entry_beta.get())
        delta = float(entry_delta.get())
        gamma = float(entry_gamma.get())
        x0 = float(entry_x0.get())
        y0 = float(entry_y0.get())
        t_end = float(entry_t_end.get())
        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, 1000)
        choice = var.get()

        if choice == "Basic Simulation":
            simulate_and_plot(alpha, beta, delta, gamma, [x0, y0], t_span, t_eval, plot_phase_plane=True)
        elif choice == "Bifurcation Analysis":
            bifurcation_analysis([x0, y0], t_span, t_eval)
        elif choice == "Stochastic Simulation":
            noise_scale = float(entry_noise.get())
            simulate_stochastic(alpha, beta, delta, gamma, [x0, y0], t_span, t_eval, noise_scale)
        elif choice == "Sensitivity Analysis":
            sensitivity_analysis(alpha, beta, delta, gamma)
        elif choice == "Time Series Analysis":
            solution = solve_ivp(lotka_volterra, t_span, [x0, y0], args=(alpha, beta, delta, gamma), t_eval=t_eval, method='RK45')
            analyze_time_series(solution.y[0], solution.y[1], t_eval)
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")
        
def interpret_sensitivity(eigenvalues):
    print("\n--- Sensitivity Analysis Interpretation ---")
    print("The eigenvalues of the Jacobian matrix help determine the stability of the system.")
    if np.all(np.real(eigenvalues) < 0):
        print("All real parts of the eigenvalues are negative, indicating that the system is stable at the equilibrium.")
    else:
        print("At least one eigenvalue has a positive real part, indicating that the equilibrium is unstable.")


root = Tk()
root.title("Lotka-Volterra Model")

Label(root, text="Prey Birth Rate (α):").grid(row=0, column=0, padx=10, pady=5)
entry_alpha = Entry(root)
entry_alpha.grid(row=0, column=1)

Label(root, text="Predation Rate (β):").grid(row=1, column=0, padx=10, pady=5)
entry_beta = Entry(root)
entry_beta.grid(row=1, column=1)

Label(root, text="Predator Reproduction Rate (δ):").grid(row=2, column=0, padx=10, pady=5)
entry_delta = Entry(root)
entry_delta.grid(row=2, column=1)

Label(root, text="Predator Death Rate (γ):").grid(row=3, column=0, padx=10, pady=5)
entry_gamma = Entry(root)
entry_gamma.grid(row=3, column=1)

Label(root, text="Initial Prey Population (x0):").grid(row=4, column=0, padx=10, pady=5)
entry_x0 = Entry(root)
entry_x0.grid(row=4, column=1)

Label(root, text="Initial Predator Population (y0):").grid(row=5, column=0, padx=10, pady=5)
entry_y0 = Entry(root)
entry_y0.grid(row=5, column=1)

Label(root, text="Simulation Time (t_end):").grid(row=6, column=0, padx=10, pady=5)
entry_t_end = Entry(root)
entry_t_end.grid(row=6, column=1)

Label(root, text="Noise Scale (for Stochastic):").grid(row=7, column=0, padx=10, pady=5)
entry_noise = Entry(root)
entry_noise.grid(row=7, column=1)

var = StringVar(root)
var.set("Basic Simulation")
options = options = ["Basic Simulation", "Bifurcation Analysis", "Stochastic Simulation", "Sensitivity Analysis", "Time Series Analysis"]
opt_menu = OptionMenu(root, var, *options)
opt_menu.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

button = Button(root, text="Run Simulation", command=run_simulation)
button.grid(row=9, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()