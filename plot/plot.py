import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
qpbenchmark_data_dir = os.path.join(script_dir, "..", "..", "qpbenchmark", "maros_meszaros_qpbenchmark", "data")
qpbenchmark_data_dir = os.path.abspath(qpbenchmark_data_dir)

qpbenchmark_names = [f[:-4] for f in os.listdir(qpbenchmark_data_dir) if f.endswith('.mat')]


problem_name = sys.argv[1] if len(sys.argv) > 1 else "None"
if (problem_name == "None" or problem_name not in qpbenchmark_names):
    raise ValueError("Input should be a valid Maros Meszaros problem name")

plot_type = sys.argv[2] if len(sys.argv) > 2 else '1'

script_dir = os.path.dirname(os.path.abspath(__file__))

if (plot_type == '1'):
    csv_file = os.path.join(script_dir, "residuals.csv")
elif (plot_type == '2'):
    csv_file = os.path.join(script_dir, "residuals.csv")
    csv_file_original = os.path.join(script_dir, "osqp_default_residuals.csv")
else:
    raise ValueError("Input should be nothing, 1, or 2")

if not os.path.exists(csv_file):
    raise FileNotFoundError("Error: File 'residuals.csv' not found.")

if (plot_type == '2' and not os.path.exists(csv_file_original)):
    raise FileNotFoundError("Error: File 'osqp_default_residuals.csv' not found.")

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

if (plot_type == '2'):
    try:
        data_original = pd.read_csv(csv_file_original)
    except FileNotFoundError:
        raise FileNotFoundError("CSV file 'osqp_default_residuals.csv' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("CSV file is malformed or could not be parsed.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading CSV file: {e}")

if (plot_type == '2'):
    cols_to_convert = ['iteration', 'prim_res', 'dual_res', 'duality_gap', 'restart']
    data_original[cols_to_convert] = data_original[cols_to_convert].apply(pd.to_numeric, errors='coerce')


required_cols = ['iteration', 'prim_res', 'dual_res', 'duality_gap', 'restart']
required_cols_original = ['iteration', 'prob_name', 'prim_res', 'dual_res', 'duality_gap', 'restart']

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise KeyError("Error: Missing columns in CSV file: 'residuals.csv'.")

if (plot_type == '2'):
    missing_cols = [col for col in required_cols if col not in data_original.columns]
    if missing_cols:
        raise KeyError("Error: Missing columns in CSV file: 'osqp_default_residuals.csv'.")

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

if (plot_type == '2'):
    (line_original,) = ax1.semilogy(
        data_original[data_original['prob_name'] == problem_name]['iteration'], data_original[data_original['prob_name'] == problem_name]['prim_res'], label="OSQP"
    )

if (plot_type == '2'):
    mask = (data_original['restart'] != 0) & (data_original['prob_name'] == problem_name)
    restart_iters_original = data_original.loc[mask, 'iteration']
    restart_values_original = data_original.loc[mask, 'prim_res']


    ax1.scatter(
        restart_iters_original,
        restart_values_original,
        color=line_original.get_color(),
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

if (plot_type == '2'):
    (line_original,) = ax2.semilogy(
        data_original[data_original['prob_name'] == problem_name]['iteration'], data_original[data_original['prob_name'] == problem_name]['dual_res'], label="OSQP"
    )

if (plot_type == '2'):
    mask = (data_original['restart'] != 0) & (data_original['prob_name'] == problem_name)
    restart_iters_original = data_original.loc[mask, 'iteration']
    restart_values_original = data_original.loc[mask, 'dual_res']


    ax2.scatter(
        restart_iters_original,
        restart_values_original,
        color=line_original.get_color(),
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

if (plot_type == '2'):
    (line_original,) = ax3.semilogy(
        data_original[data_original['prob_name'] == problem_name]['iteration'], data_original[data_original['prob_name'] == problem_name]['duality_gap'].abs(), label="OSQP"
    )

if (plot_type == '2'):
    mask = (data_original['restart'] != 0) & (data_original['prob_name'] == problem_name)
    restart_iters_original = data_original.loc[mask, 'iteration']
    restart_values_original = data_original.loc[mask, 'duality_gap'].abs()


    ax3.scatter(
        restart_iters_original,
        restart_values_original,
        color=line_original.get_color(),
        marker="o",
        s=50,
        alpha=0.7,
    )

ax3.set_ylabel("Duality Gap")
ax3.grid(True, which="both", ls="-", alpha=0.5)
ax3.legend()


plt.tight_layout()

save_file = os.path.join(script_dir, "convergence_plot.pdf")
print("save_file: ", save_file)
plt.savefig(save_file)
print("Convergence results plot saved to 'convergence_plot.pdf'")
