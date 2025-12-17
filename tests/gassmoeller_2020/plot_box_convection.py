import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlim(1.5e8, 4e8)
ax.set_ylim(2e5, 1.2e6)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Top heat flux (W/m)")
ax.grid(which="both")

for apx in ["BA", "EBA", "TALA", "ALA", "ICA", "HCA", "PDA"]:
    diags = np.load(f"diags_box_convection_{apx}.npz")
    ax.plot(diags["time"], diags["heat_flux_top"], label=apx)

plt.legend()
plt.savefig("box_convection.pdf", bbox_inches="tight", dpi=300)
