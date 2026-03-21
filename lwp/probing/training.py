"""Probe training: ridge regression from activations to targets.

Linear probes (RidgeCV) with episode-level cross-validation.
The standard approach for representation probing (Alain & Bengio 2017).

See probing-tooling.md (Section 3) for the full specification.
"""
from collections import Counter

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from lwp.probing.targets import (
    PARAMETRIC_TARGET_NAMES,
    BEHAVIORAL_TARGET_NAMES,
    KINEMATIC_TARGET_NAMES,
    ALL_TARGET_NAMES,
)

# Default ridge alphas for RidgeCV — spans 4 orders of magnitude,
# letting the model automatically pick the best regularization strength
# via efficient leave-one-out cross-validation within each fold.
_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Number of top units to report — these are the hidden units whose
# regression weights have the largest absolute values, indicating
# which neurons contribute most to predicting each target.
_N_TOP_UNITS = 5


def _episode_kfold(episode_ids: np.ndarray, n_folds: int = 5, seed: int = 42):
    """Generate episode-level k-fold splits.

    Unlike standard k-fold which splits individual timesteps, this groups
    all timesteps from the same episode together. This prevents data leakage:
    consecutive timesteps within an episode are highly correlated, so
    splitting them across train/test would inflate R² scores.

    Yields (train_mask, test_mask) boolean arrays over the timestep dimension.
    """
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episodes)

    fold_size = len(unique_episodes) // n_folds
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            # Last fold gets any remainder episodes
            test_episodes = set(unique_episodes[start:])
        else:
            test_episodes = set(unique_episodes[start:start + fold_size])

        test_mask = np.isin(episode_ids, list(test_episodes))
        train_mask = ~test_mask
        yield train_mask, test_mask


def train_single_probe(
    activations: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    n_folds: int = 5,
    n_top: int = _N_TOP_UNITS,
) -> dict:
    """Train a linear probe (ridge regression) with episode-level CV.

    The probe tests whether a target variable (e.g. gravity, TWR) can be
    linearly decoded from the network's hidden activations. High R² means
    the network has learned a representation that linearly encodes that
    quantity; low R² means it hasn't (or encodes it nonlinearly).

    Args:
        activations: (N, hidden_dim) activation matrix from one layer.
        target: (N,) target vector (single scalar target per timestep).
        episode_ids: (N,) episode IDs for episode-level CV splitting.
        n_folds: Number of CV folds (episodes shuffled, then split).
        n_top: Number of top contributing units to report.

    Returns:
        Dict with:
            r2_mean, r2_std, r2_folds: Cross-validated R² statistics.
            alpha: Most commonly selected regularization strength.
            top_units: Indices of top contributing hidden units.
            top_weights: Corresponding regression weights (signed).
            coefficients: Full weight vector (hidden_dim,) from final fit.
            intercept: Scalar intercept from final fit on all data.
    """
    r2_folds = []
    alphas_selected = []

    for train_mask, test_mask in _episode_kfold(episode_ids, n_folds):
        X_train, X_test = activations[train_mask], activations[test_mask]
        y_train, y_test = target[train_mask], target[test_mask]

        # Standardize target (fit on train, transform both) so that
        # R² is comparable across targets with different scales.
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()

        # Fit ridge with automatic alpha selection via efficient LOO-CV
        ridge = RidgeCV(alphas=_ALPHAS)
        ridge.fit(X_train, y_train_scaled)

        r2 = ridge.score(X_test, y_test_scaled)
        r2_folds.append(float(r2))
        alphas_selected.append(float(ridge.alpha_))

    # Fit final model on ALL data to get stable weight estimates for
    # top_units identification and coefficient persistence (Phase C needs these).
    scaler_final = StandardScaler()
    y_scaled = scaler_final.fit_transform(target.reshape(-1, 1)).ravel()
    ridge_final = RidgeCV(alphas=_ALPHAS)
    ridge_final.fit(activations, y_scaled)

    # Identify top units by absolute weight magnitude — these are the
    # hidden neurons most important for linearly predicting this target.
    weights = ridge_final.coef_
    top_indices = np.argsort(np.abs(weights))[::-1][:n_top]
    top_units = top_indices.tolist()
    top_weights = [round(float(weights[i]), 4) for i in top_indices]

    # Report the most commonly selected alpha across CV folds
    alpha_counts = Counter(alphas_selected)
    alpha = alpha_counts.most_common(1)[0][0]

    return {
        "r2_mean": round(float(np.mean(r2_folds)), 4),
        "r2_std": round(float(np.std(r2_folds)), 4),
        "r2_folds": [round(r, 4) for r in r2_folds],
        "alpha": alpha,
        "top_units": top_units,
        "top_weights": top_weights,
        "coefficients": weights.astype(np.float32),
        "intercept": float(ridge_final.intercept_),
    }


def train_all_probes(
    data: dict,
    layers: list[str] | None = None,
    targets: list[str] | None = None,
    n_folds: int = 5,
) -> tuple[dict, dict]:
    """Train probes for all layer x target combinations.

    Iterates over every (layer, target) pair and trains a separate
    ridge regression probe for each. This produces the full "probe matrix":
    rows = layers (L1, L2), columns = targets (7 parametric + 5 behavioral).

    Args:
        data: Dict with activations_L1, activations_L2, physics_params,
            behavioral, episode_ids (as from collect_probe_data / np.load).
        layers: Which layers to probe. Default: ["L1", "L2"].
        targets: Which targets to probe. Default: all 12.
        n_folds: CV folds.

    Returns:
        Tuple of (probes_dict, coefficients_dict):
            probes_dict: Nested {layer: {target: {r2_mean, r2_std, ...}}}
                (coefficients/intercept excluded — they go in coefficients_dict)
            coefficients_dict: Flat {"{layer}/{target}": np.ndarray(hidden_dim,)}
                keyed for easy np.savez(**coefficients_dict).
                Also includes "{layer}/{target}_intercept" scalar entries.
    """
    if layers is None:
        # Auto-detect layers from data keys (activations_CNN, activations_L1, etc.)
        # or fall back to metadata layer_names if present.
        if "layer_names" in data:
            import json
            layer_names_raw = data["layer_names"]
            # np.load returns numpy strings; decode if needed
            if hasattr(layer_names_raw, 'item'):
                layer_names_raw = str(layer_names_raw.item())
            else:
                layer_names_raw = str(layer_names_raw)
            layers = json.loads(layer_names_raw)
        else:
            # Fallback: scan for activations_* keys
            prefix = "activations_"
            layers = [k[len(prefix):] for k in data if k.startswith(prefix)]
            if not layers:
                layers = ["L1", "L2"]
    if targets is None:
        targets = list(ALL_TARGET_NAMES)
        # Auto-include kinematic targets when data has them
        if "kinematic" in data:
            targets = targets + list(KINEMATIC_TARGET_NAMES)

    episode_ids = data["episode_ids"]

    # Build target vectors from the data arrays.
    # Parametric: 7 raw physics config values (constant per episode)
    # Behavioral: 5 derived quantities like TWR (constant per episode)
    # Kinematic: 8 state variables (vary per timestep) — x, y, vx, vy, etc.
    target_arrays = {}
    for i, name in enumerate(PARAMETRIC_TARGET_NAMES):
        target_arrays[name] = data["physics_params"][:, i]
    for i, name in enumerate(BEHAVIORAL_TARGET_NAMES):
        target_arrays[name] = data["behavioral"][:, i]
    if "kinematic" in data:
        for i, name in enumerate(KINEMATIC_TARGET_NAMES):
            target_arrays[name] = data["kinematic"][:, i]

    total_probes = len(layers) * len(targets)
    probe_idx = 0

    results = {}
    coefficients = {}
    for layer in layers:
        key = f"activations_{layer}"
        if key not in data:
            print(f"  Skipping layer {layer} (no {key} in data)")
            continue
        activations = data[key]
        print(f"\n  Layer {layer} ({activations.shape[1]}D):")
        results[layer] = {}

        for target_name in targets:
            if target_name not in target_arrays:
                continue
            probe_idx += 1
            target = target_arrays[target_name]
            print(f"    [{probe_idx}/{total_probes}] {layer}/{target_name}...", end="", flush=True)
            probe_result = train_single_probe(
                activations, target, episode_ids, n_folds=n_folds,
            )
            r2 = probe_result["r2_mean"]
            marker = "***" if r2 > 0.7 else "**" if r2 > 0.4 else "*" if r2 > 0.2 else ""
            print(f" R²={r2:.3f} ±{probe_result['r2_std']:.3f} {marker}")
            # Separate coefficients from JSON-serializable results —
            # R² scores go to probe_results.json, weight matrices go
            # to probe_coefficients.npz for Phase C ablation/steering.
            coefficients[f"{layer}/{target_name}"] = probe_result.pop("coefficients")
            coefficients[f"{layer}/{target_name}_intercept"] = np.float32(
                probe_result.pop("intercept")
            )
            results[layer][target_name] = probe_result

    return results, coefficients
