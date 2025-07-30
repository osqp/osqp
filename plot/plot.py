import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "residuals.csv")

if not os.path.exists(csv_file):
    raise FileNotFoundError("Error: File 'residuals.csv' not found.")

try:
    data = pd.read_csv(csv_file)
except FileNotFoundError:
    raise FileNotFoundError("CSV file 'residuals.csv' not found.")
except pd.errors.EmptyDataError:
    raise ValueError("CSV file is empty.")
except pd.errors.ParserError:
    raise ValueError("CSV file is malformed or could not be parsed.")
except Exception as e:
    raise RuntimeError(f"Unexpected error reading CSV file: {e}")

required_cols = ['iteration', 'prim_res', 'dual_res', 'duality_gap', 'restart']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise KeyError("Error: Missing columns in CSV file: 'residuals.csv'.")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot primal residual
(line,) = ax1.semilogy(
    data['iteration'], data['prim_res'], label="OSQP With Reflected Halpern Restarts"
)

# Mark restart points
restart_iters = data[data['restart'] != 0]['iteration']
restart_values = data[data['restart'] != 0]['prim_res']

ax1.scatter(
    restart_iters,
    restart_values,
    color=line.get_color(),
    marker="o",
    s=50,
    alpha=0.7,
)

ax1.set_ylabel("Primal Residual")
ax1.grid(True, which="both", ls="-", alpha=0.5)
ax1.legend()


# Plot dual residual
(line,) = ax2.semilogy(
    data['iteration'], data['dual_res'], label="OSQP With Reflected Halpern Restarts"
)

# Mark restart points
restart_iters = data[data['restart'] != 0]['iteration']
restart_values = data[data['restart'] != 0]['dual_res']

ax2.scatter(
    restart_iters,
    restart_values,
    color=line.get_color(),
    marker="o",
    s=50,
    alpha=0.7,
)

ax2.set_ylabel("Dual Residual")
ax2.grid(True, which="both", ls="-", alpha=0.5)
ax2.legend()


# Plot duality gap
(line,) = ax3.semilogy(
    data['iteration'], data['duality_gap'].abs(), label="OSQP With Reflected Halpern Restarts"
)

# Mark restart points
restart_iters = data[data['restart'] != 0]['iteration']
restart_values = data[data['restart'] != 0]['duality_gap'].abs()

ax3.scatter(
    restart_iters,
    restart_values,
    color=line.get_color(),
    marker="o",
    s=50,
    alpha=0.7,
)

ax3.set_ylabel("Duality Gap")
ax3.grid(True, which="both", ls="-", alpha=0.5)
ax3.legend()


plt.tight_layout()

plt.savefig("../plot/convergence_plot.pdf")
print("Convergence results plot saved to 'convergence_plot.pdf'")
