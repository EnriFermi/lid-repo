"""Plot existing canary evidence; never run training or choose a policy."""

from pathlib import Path
import json

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="spiral_spaghetti_convergence")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.output).resolve()
    styles = {
        "base": ("#777777", "8k checkpoint", ":"),
        "same_lr": ("#bd3d3a", "10k: LR 2e-4", "-"),
        "same_lr_average": ("#d88432", "10k: LR 2e-4 + average", "--"),
        "lower_lr": ("#235eb5", "10k: LR 2e-5", "-"),
        "lower_lr_average": ("#248158", "10k: LR 2e-5 + average", "--"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, dataset in enumerate(cfg.datasets):
        directory = root / str(dataset)
        history = json.loads((directory / "history.json").read_text())
        results = json.loads((directory / "results.json").read_text())
        for stage in ("base", "same_lr", "lower_lr"):
            entries = [entry for entry in history if entry["stage"] == stage]
            color, label, style = styles[stage]
            axes[row, 0].semilogy([entry["step"] for entry in entries],
                                  [entry["mean_grid_risk"] for entry in entries],
                                  color=color, linestyle=style, marker=".", label=label)
        axes[row, 0].axvline(8000, color="black", linewidth=.8, alpha=.4)
        axes[row, 0].set(xlabel="Optimizer updates", ylabel="Held-out grid-mean native risk")
        for column, readout in ((1, "response"), (2, "full")):
            for entry in results:
                if entry["readout"] != readout:
                    continue
                color, label, style = styles[entry["stage"]]
                axes[row, column].plot(list(cfg.scales), entry["selection_mae_curve"],
                                       color=color, linestyle=style, label=label)
            axes[row, column].set_xscale("log")
            axes[row, column].set(xlabel="Physical noise scale lambda",
                                   ylabel=f"Held-out LID MAE: {readout}")
            if readout == "full":
                axes[row, column].set_yscale("log")
            else:
                axes[row, column].axhline(0, color="black", linewidth=.8, alpha=.5)
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.set_title("Spiral" if "spiral" in str(dataset) else "Spaghetti")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Exact rendered benchmark data: lower denoising loss does not ensure lower LID error")
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(root / "convergence_overview.png", dpi=160)
    fig.savefig(root / "convergence_overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
