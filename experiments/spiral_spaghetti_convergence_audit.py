"""Full-test verification at previously frozen canary scale choices.

No checkpoint, readout or scale is selected using the expanded test sample.
Paired bootstrap describes query uncertainty only, not training uncertainty.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from experiments.spiral_spaghetti_convergence import exact_trace, write_json
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


@hydra.main(version_base=None, config_path="../configs", config_name="spiral_spaghetti_convergence")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(cfg.output).resolve()
    deadline = datetime.fromisoformat(str(cfg.deadline_utc)).timestamp()
    results = []
    for name in cfg.datasets:
        directory = root / str(name)
        rows = json.loads((directory / "results.json").read_text())
        base = torch.load(directory / "base.pt", map_location="cpu", weights_only=False)
        test_raw = np.load(root / "exact_subset/benchmarks" / str(name) / "test/dataset.npy")
        targets = np.load(root / "exact_subset/benchmarks" / str(name) / "test/lid.npy")
        if not np.all(targets == 1):
            raise ValueError("Expected archived test LID=1")
        raw = torch.from_numpy(test_raw.reshape(len(test_raw), -1).astype(np.float32))
        queries = ((raw - base["mean"]) / base["rms"]).to(str(cfg.device))
        model = ScaleConditionedNeuralField(NeuralFieldConfig(
            ambient_dim=queries.shape[1], hidden_dim=int(cfg.hidden_dim),
            depth=int(cfg.depth), condition_dim=int(cfg.condition_dim),
            zero_init_output=True,
        )).to(str(cfg.device))
        saved_predictions = {}
        for stage in ("base", "same_lr", "same_lr_average", "lower_lr", "lower_lr_average"):
            payload = torch.load(directory / f"{stage}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state"])
            model.eval()
            scale_rows = [row for row in rows if row["stage"] == stage]
            cache = {}
            for row in scale_rows:
                lam = row["selected_scale"]
                if lam not in cache:
                    value, response = exact_trace(model, queries, lam)
                    correction = ((value.double() - queries.double()) / lam).square().sum(-1)
                    cache[lam] = {"response": response.double().cpu().numpy(),
                                  "full": (response.double() + correction).cpu().numpy()}
                predictions = cache[lam][row["readout"]]
                previous = np.load(directory / f"{stage}_predictions.npz")
                scale_index = list(cfg.scales).index(lam)
                np.testing.assert_allclose(
                    predictions[:int(cfg.query_count)],
                    previous[f"test_{row['readout']}"][scale_index],
                    rtol=1e-6, atol=1e-6,
                )
                key = f"{stage}_{row['readout']}"
                saved_predictions[key] = predictions
                mae = float(np.abs(predictions - 1).mean())
                entry = {"dataset": str(name), "stage": stage,
                         "readout": row["readout"], "scale": lam,
                         "test_count": len(queries), "test_mae": mae,
                         "mean_lid": float(predictions.mean()),
                         "median_lid": float(np.median(predictions)),
                         "selection_unchanged": True}
                results.append(entry)
                print(json.dumps(entry), flush=True)
                write_json(root / "full_test_audit.json", results)
                np.savez(directory / "full_test_audit_predictions.npz", **saved_predictions)
                if time.time() > deadline:
                    raise TimeoutError("Time box reached; partial audit preserved")
        pairs = []
        rng = np.random.default_rng(int(cfg.seed) + 401)
        bootstrap_ids = rng.integers(0, len(queries), (2000, len(queries)))
        for readout in ("response", "full"):
            control = np.abs(saved_predictions[f"same_lr_{readout}"] - 1)
            for stage in ("lower_lr", "same_lr_average", "lower_lr_average"):
                delta = np.abs(saved_predictions[f"{stage}_{readout}"] - 1) - control
                means = delta[bootstrap_ids].mean(1)
                pairs.append({"readout": readout, "stage": stage,
                              "delta_mae_vs_same_lr": float(delta.mean()),
                              "paired_query_bootstrap_95": np.quantile(means, [.025, .975]).tolist(),
                              "scope": "fixed policies/scales; query uncertainty only"})
        write_json(directory / "paired_query_audit.json", pairs)
    print("FULL_TEST_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
