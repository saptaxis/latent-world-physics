"""Probe training: linear and nonlinear probes from activations to targets.

Linear probes (RidgeCV) and MLP probes (MLPRegressor) with episode-level
cross-validation. Linear probes test linear decodability (Alain & Bengio 2017).
MLP probes test nonlinear accessibility — comparing linear vs MLP R² reveals
whether information is linearly encoded or requires nonlinear readout.

See probing-tooling.md (Section 3) for the full specification.
"""
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
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


def train_single_mlp_probe(
    activations: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    n_folds: int = 5,
    hidden_sizes: tuple[int, ...] = (64,),
    max_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Train a nonlinear probe (MLP) with episode-level CV.

    Tests whether a target variable can be decoded from activations via
    a nonlinear map. Comparing MLP R² to linear probe R² reveals whether
    information is linearly accessible or nonlinearly encoded.

    Automatically selects solver based on dataset size:
    - lbfgs for small datasets (<10K samples): faster, more precise
    - adam with early stopping for large datasets: scales to 100K+ samples

    Both input features and target are standardized per fold to make R²
    comparable across targets with different scales.

    Args:
        activations: (N, hidden_dim) activation matrix from one layer.
        target: (N,) target vector (single scalar target per timestep).
        episode_ids: (N,) episode IDs for episode-level CV splitting.
        n_folds: Number of CV folds (episodes shuffled, then split).
        hidden_sizes: MLP hidden layer sizes. Default (64,) = one hidden
            layer with 64 units. Universal approximation theorem guarantees
            this can learn any continuous nonlinear map with enough units.
        max_iter: Maximum training iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
            r2_mean, r2_std, r2_folds: Cross-validated R² statistics.
            hidden_sizes: Architecture used (for documentation).
            probe_type: "mlp" (self-describing).
    """
    r2_folds = []

    for train_mask, test_mask in _episode_kfold(episode_ids, n_folds):
        X_train, X_test = activations[train_mask], activations[test_mask]
        y_train, y_test = target[train_mask], target[test_mask]

        # Scale input features — critical for MLP convergence.
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Standardize target so R² is comparable across targets with different scales.
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # lbfgs is faster/more precise for small datasets but fails on large
        # ones (stores full Hessian approximation, O(n²) memory).
        # adam with early stopping scales to 100K+ samples.
        n_train = X_train_scaled.shape[0]
        if n_train < 10_000:
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_sizes,
                solver="lbfgs",
                alpha=0.1,
                max_iter=max_iter,
                random_state=seed,
            )
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_sizes,
                solver="adam",
                alpha=0.0001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                max_iter=max_iter,
                random_state=seed,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            mlp.fit(X_train_scaled, y_train_scaled)

        r2 = mlp.score(X_test_scaled, y_test_scaled)
        r2_folds.append(float(r2))

    return {
        "r2_mean": round(float(np.mean(r2_folds)), 4),
        "r2_std": round(float(np.std(r2_folds)), 4),
        "r2_folds": [round(r, 4) for r in r2_folds],
        "hidden_sizes": hidden_sizes,
        "probe_type": "mlp",
    }


def train_all_probes(
    data: dict,
    layers: list[str] | None = None,
    targets: list[str] | None = None,
    n_folds: int = 5,
    probe_types: tuple[str, ...] = ("linear",),
    mlp_hidden_sizes: tuple[int, ...] = (64,),
) -> tuple[dict, dict]:
    """Train probes for all layer x target combinations.

    Iterates over every (layer, target, probe_type) triple and trains a
    separate probe for each. Supports linear probes (RidgeCV), MLP probes
    (MLPRegressor), or both in a single call.

    Args:
        data: Dict with activations_L1, activations_L2, physics_params,
            behavioral, episode_ids (as from collect_probe_data / np.load).
        layers: Which layers to probe. Default: auto-detect from data keys.
        targets: Which targets to probe. Default: all 12 (+ kinematic if present).
        n_folds: CV folds.
        probe_types: Which probe types to run. Default: ("linear",).
            Pass ("linear", "mlp") to run both. When multiple types are
            requested, results are nested by probe type at the top level.
        mlp_hidden_sizes: MLP hidden layer sizes (only used when "mlp"
            is in probe_types). Default: (64,).

    Returns:
        Tuple of (probes_dict, coefficients_dict):

        When probe_types has ONE entry (backward compatible):
            probes_dict: {layer: {target: {r2_mean, r2_std, ...}}}
            coefficients_dict: {"{layer}/{target}": np.ndarray} (linear only)

        When probe_types has MULTIPLE entries:
            probes_dict: {probe_type: {layer: {target: {r2_mean, ...}}}}
            coefficients_dict: {"{layer}/{target}": np.ndarray} (linear only)
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

    total_probes = len(layers) * len(targets) * len(probe_types)
    probe_idx = 0
    multi_type = len(probe_types) > 1

    # When multiple probe types requested, nest results by type.
    # When single type, keep flat structure for backward compatibility.
    if multi_type:
        results = {pt: {} for pt in probe_types}
    else:
        results = {}
    coefficients = {}

    for layer in layers:
        key = f"activations_{layer}"
        if key not in data:
            print(f"  Skipping layer {layer} (no {key} in data)")
            continue
        activations = data[key]
        print(f"\n  Layer {layer} ({activations.shape[1]}D):")

        if multi_type:
            for pt in probe_types:
                results[pt][layer] = {}
        else:
            results[layer] = {}

        for target_name in targets:
            if target_name not in target_arrays:
                continue

            target = target_arrays[target_name]

            for probe_type in probe_types:
                probe_idx += 1
                label = f"{probe_type} " if multi_type else ""
                print(
                    f"    [{probe_idx}/{total_probes}] {label}{layer}/{target_name}...",
                    end="", flush=True,
                )

                if probe_type == "linear":
                    probe_result = train_single_probe(
                        activations, target, episode_ids, n_folds=n_folds,
                    )
                    # Separate coefficients from JSON-serializable results —
                    # R² scores go to probe_results.json, weight matrices go
                    # to probe_coefficients.npz for Phase C ablation/steering.
                    coefficients[f"{layer}/{target_name}"] = probe_result.pop(
                        "coefficients"
                    )
                    coefficients[f"{layer}/{target_name}_intercept"] = np.float32(
                        probe_result.pop("intercept")
                    )
                elif probe_type == "mlp":
                    probe_result = train_single_mlp_probe(
                        activations, target, episode_ids,
                        n_folds=n_folds, hidden_sizes=mlp_hidden_sizes,
                    )
                else:
                    raise ValueError(f"Unknown probe_type: {probe_type}")

                r2 = probe_result["r2_mean"]
                marker = (
                    "***" if r2 > 0.7
                    else "**" if r2 > 0.4
                    else "*" if r2 > 0.2
                    else ""
                )
                print(f" R²={r2:.3f} ±{probe_result['r2_std']:.3f} {marker}")

                if multi_type:
                    results[probe_type][layer][target_name] = probe_result
                else:
                    results[layer][target_name] = probe_result

    return results, coefficients
