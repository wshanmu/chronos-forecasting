"""Prepare invariant UWB link-geometry embeddings from desk tap estimates.

For each deployment, this script:
    1. loads optimized node/desk coordinates from Evaluate_visualization,
    2. estimates desk XYZ locations from estimated_tap_dict,
    3. prepares raw_geometry with shape [desk, link, feature],
    4. saves CSV/NPZ feature outputs and a 3D comparison plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    pass

node_coordinate_dict = {
    "Deployment7": {
        "node0": [0.0, 0.0, 2.7091626268259756],
        "node1": [4.725982484856591, 0.0, 2.640675412182338],
        "node2": [-0.02671092883830918, 6.673281161124579, 2.7161425508561736],
        "node3": [4.733056019563274, 6.65119227859373, 2.734019410135513],
    },
    "Deployment8": {
        "node0": [0.0, 0.0, 2.7344675600271153],
        "node1": [6.372149194795343, 0.0, 2.6965972871159236],
        "node2": [0.07368426857463686, 6.963806633805904, 2.7344675781640153],
        "node3": [6.532057351563105, 6.961557269746283, 2.6344675746929465],
    },
    "Deployment9": {
        "node0": [0.0, 0.0, 2.6500000000179367],
        "node1": [6.522806384048381, 0.0, 2.649999999981966],
        "node2": [-0.07146784067043792, 5.117680883599404, 2.7500000000021307],
        "node3": [6.276476075307203, 5.340532437146415, 2.7499999999979674],
    },
    "Deployment10": {
        "node0": [0.0, 0.0, 2.6621510406046265],
        "node1": [6.524980969026581, 0.0, 2.662151040637111],
        "node2": [-0.04434393365003826, 5.116011845716329, 2.713546878136202],
        "node3": [6.299971670610867, 5.342935995029722, 2.7621510406220606],
    },
}

estimated_tap_dict = {
    "Deployment7": {
        '1': [20, 25, 42, 12], #
        '2': [20, 13, 41, 22], #
        '3': [33, 25, 28, 11], #
        '4': [31, 12, 28, 22], #
        '5': [49, 25, 16, 14], # 
        '6': [48, 13, 16, 25], #
    },
    "Deployment8": {
        '1': [13, 30, 44, 20], 
        '2': [13, 20, 43, 29], #
        '3': [24, 29, 28, 16], #
        '4': [23, 18, 29, 27], # 
        '5': [39, 30, 16, 18], # 
        '6': [40, 19, 15, 29], # 
    },
    "Deployment9": {
        '1': [12, 37, 27, 19],
        '2': [12, 22, 26, 32],
        '3': [25, 36, 14, 19],
        '4': [24, 21, 14, 33]
    },
    "Deployment10": {
        '1': [23, 33, 14, 21],
        '2': [12, 34, 27, 21],
        '3': [24, 20, 14, 32],
        '4': [12, 20, 28, 35]
    }
}


# --- Geometry/tap configuration ------------------------------------------
# estimated_tap_dict order, exactly as provided:
#   node0-1, node1-3, node2-3, node0-2
TAP_PAIR_ORDER = ((0, 1), (1, 3), (2, 3), (0, 2))
TAP_TO_EXCESS_LENGTH_M = 0.3002 / 2.0
RAW_GEOMETRY_COLUMNS = (
    "baseline_m",
    "excess_path_m",
    "cos_beta",
    "d_parallel_m",
    "d_perp_m",
)

DEPLOYMENT_TO_ROOM_NAME = {
    "Deployment7": "Room_1_Deployment_7_Conference",
    "Deployment8": "Room_2_Deployment_8_Keysight",
    "Deployment9": "Room_3_Deployment_9_Boelter",
    "Deployment10": "Room_3_Deployment_10_Boelter",
}

DESK_HEIGHT_PRIOR_BY_DEPLOYMENT_M = {
    "Deployment7": 0.713,
    "Deployment8": 0.728,
    "Deployment9": 0.75,
    "Deployment10": 0.75,
}

DESKPULSE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DESKPULSE_DIR / "Evaluate_visualization"

FLIP_X_AXIS_IN_PLOT = True
PLOT_XY_MARGIN_M = 0.35
PLOT_Z_MARGIN_M = 0.15

DEFAULT_GEOMETRY_FEATURE_LOOKUP: dict[str, dict[int, np.ndarray]] | None = None


@dataclass(frozen=True)
class GeometryEmbedding:
    deployment: str
    desk_names: list[str]
    link_index: np.ndarray
    scene_xyz: np.ndarray
    optimized_scene_xyz: np.ndarray
    tap_index: np.ndarray
    raw_geometry: np.ndarray
    fitted_raw_geometry: np.ndarray
    excess_path_m: np.ndarray
    fitted_excess_path_m: np.ndarray
    reference_bins: np.ndarray


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_reference_bins(
    excess_path_m: np.ndarray,
    *,
    range_resolution_m: float,
    los_bin: int,
) -> np.ndarray:
    if range_resolution_m <= 0.0:
        raise ValueError(f"range_resolution_m must be positive, found {range_resolution_m}")
    return float(los_bin) + excess_path_m / float(range_resolution_m)


def compute_symmetric_link_geometry(
    *,
    scene_xyz: np.ndarray,
    node_xyz: np.ndarray,
    link_index: np.ndarray,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute invariant geometry [b, rho, cos_beta, d_parallel, d_perp].

    The returned raw_geometry has shape [P, L, 5], where P is scene points and
    L is links. The feature order matches RAW_GEOMETRY_COLUMNS.
    """
    scene_xyz = np.asarray(scene_xyz, dtype=float)
    node_xyz = np.asarray(node_xyz, dtype=float)
    link_index = np.asarray(link_index, dtype=int)

    if scene_xyz.ndim != 2 or scene_xyz.shape[-1] != 3:
        raise ValueError(f"scene_xyz must have shape [P, 3], found {scene_xyz.shape}")
    if node_xyz.ndim != 2 or node_xyz.shape[-1] != 3:
        raise ValueError(f"node_xyz must have shape [N, 3], found {node_xyz.shape}")
    if link_index.ndim != 2 or link_index.shape[-1] != 2:
        raise ValueError(f"link_index must have shape [L, 2], found {link_index.shape}")
    if link_index.size == 0:
        raise ValueError("link_index must contain at least one link")
    if int(link_index.min()) < 0 or int(link_index.max()) >= node_xyz.shape[0]:
        raise ValueError("link_index contains a node index outside node_xyz")

    endpoints_a = node_xyz[link_index[:, 0]]
    endpoints_b = node_xyz[link_index[:, 1]]
    baseline_vec = endpoints_b - endpoints_a
    baseline = np.linalg.norm(baseline_vec, axis=-1)
    if np.any(baseline <= eps):
        raise ValueError("link_index contains a degenerate zero-length link")

    p = scene_xyz[:, None, :]
    a = endpoints_a[None, :, :]
    b = endpoints_b[None, :, :]

    vec_pa = a - p
    vec_pb = b - p
    d_a = np.linalg.norm(vec_pa, axis=-1)
    d_b = np.linalg.norm(vec_pb, axis=-1)

    excess_path = d_a + d_b - baseline[None, :]
    cos_beta = np.sum(vec_pa * vec_pb, axis=-1) / (np.maximum(d_a, eps) * np.maximum(d_b, eps))
    cos_beta = np.clip(cos_beta, -1.0, 1.0)

    midpoint = 0.5 * (a + b)
    unit_baseline = baseline_vec[None, :, :] / baseline[None, :, None]
    rel_midpoint = p - midpoint
    signed_parallel = np.sum(rel_midpoint * unit_baseline, axis=-1)
    d_parallel = np.abs(signed_parallel)
    perpendicular_vec = rel_midpoint - signed_parallel[..., None] * unit_baseline
    d_perp = np.linalg.norm(perpendicular_vec, axis=-1)

    raw_geometry = np.stack(
        [
            np.broadcast_to(baseline[None, :], excess_path.shape),
            excess_path,
            cos_beta,
            d_parallel,
            d_perp,
        ],
        axis=-1,
    )
    return raw_geometry, excess_path


def normalize_layout_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def layout_aliases(name: str) -> set[str]:
    normalized = normalize_layout_name(name)
    aliases = {normalized}
    for deployment, room_name in DEPLOYMENT_TO_ROOM_NAME.items():
        deployment_normalized = normalize_layout_name(deployment)
        room_normalized = normalize_layout_name(room_name)
        if normalized in {deployment_normalized, room_normalized}:
            aliases.update({deployment_normalized, room_normalized})
    return aliases


def find_layout_mapping_key(mapping: dict, layout: str):
    target_aliases = layout_aliases(layout)
    for key in mapping:
        if layout_aliases(str(key)) & target_aliases:
            return key
    raise KeyError(f"No entry found for layout {layout!r}")


def resolve_deployment_key(name: str) -> str:
    normalized = normalize_layout_name(name)
    if name in estimated_tap_dict:
        return name

    for deployment, room_name in DEPLOYMENT_TO_ROOM_NAME.items():
        deployment_normalized = normalize_layout_name(deployment)
        room_normalized = normalize_layout_name(room_name)
        if normalized in {deployment_normalized, room_normalized}:
            return deployment

    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        candidate = f"Deployment{digits}"
        if candidate in estimated_tap_dict:
            return candidate

    raise KeyError(f"Unknown deployment name: {name}")


def optimized_coordinate_csv_path(deployment: str) -> Path:
    room_name = DEPLOYMENT_TO_ROOM_NAME.get(deployment)
    if room_name:
        candidate = OUTPUT_DIR / f"node_desk_coordinate_solution_{room_name}.csv"
        if candidate.exists():
            return candidate

    digits = "".join(ch for ch in deployment if ch.isdigit())
    matches = sorted(OUTPUT_DIR.glob(f"node_desk_coordinate_solution_*Deployment_{digits}_*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No optimized coordinate CSV found for {deployment}")


def load_fallback_nodes(deployment: str) -> dict[int, np.ndarray]:
    if deployment not in node_coordinate_dict:
        return {}

    nodes = {}
    for key, coords in node_coordinate_dict[deployment].items():
        node_idx = int(key.replace("node", ""))
        nodes[node_idx] = np.asarray(coords, dtype=float)
    return nodes


def load_optimized_geometry(deployment_name: str) -> tuple[str, dict[int, np.ndarray], dict[str, np.ndarray], Path | None]:
    deployment = resolve_deployment_key(deployment_name)

    nodes: dict[int, np.ndarray] = {}
    optimized_desks: dict[str, np.ndarray] = {}
    csv_path: Path | None = None

    try:
        csv_path = optimized_coordinate_csv_path(deployment)
    except FileNotFoundError:
        nodes = load_fallback_nodes(deployment)
        if not nodes:
            raise
        return deployment, nodes, optimized_desks, None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point_name = row["point"].strip()
            coords = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            lowered = point_name.lower()
            if lowered.startswith("n"):
                nodes[int(lowered.replace("n", ""))] = coords
            elif lowered.startswith("desk"):
                optimized_desks[lowered.replace("desk", "")] = coords

    missing_nodes = [idx for idx in range(4) if idx not in nodes]
    if missing_nodes:
        raise ValueError(f"{csv_path} is missing optimized nodes: {missing_nodes}")

    return deployment, nodes, optimized_desks, csv_path


def nodes_as_array(nodes: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([nodes[idx] for idx in range(4)])


def coerce_node_xyz(node_coordinates) -> np.ndarray:
    """Convert supported node-coordinate formats to a [4, 3] array."""
    if isinstance(node_coordinates, dict):
        nodes = {}
        for key, coords in node_coordinates.items():
            if isinstance(key, int):
                node_idx = key
            else:
                clean = str(key).strip().lower().replace("node", "").replace("n", "")
                node_idx = int(clean)
            nodes[node_idx] = np.asarray(coords, dtype=float)
        missing = [idx for idx in range(4) if idx not in nodes]
        if missing:
            raise ValueError(f"Missing node coordinates for node ids: {missing}")
        return nodes_as_array(nodes)

    nodes = np.asarray(node_coordinates, dtype=float)
    if nodes.shape != (4, 3):
        raise ValueError(f"node coordinates must have shape (4, 3), found {nodes.shape}")
    return nodes


def standalone_node_xyz(layout: str) -> np.ndarray:
    """Return hardcoded optimized node coordinates for a deployment layout."""
    deployment = resolve_deployment_key(layout)
    nodes = load_fallback_nodes(deployment)
    if not nodes:
        raise KeyError(f"No standalone node coordinates are defined for {deployment}")
    return nodes_as_array(nodes)


def resolve_desk_height_prior(
    layout: str,
    nodes: np.ndarray,
    desk_height_prior_by_layout: dict | None = None,
) -> float:
    if desk_height_prior_by_layout:
        try:
            key = find_layout_mapping_key(desk_height_prior_by_layout, layout)
            return float(desk_height_prior_by_layout[key])
        except KeyError:
            pass

    try:
        deployment = resolve_deployment_key(layout)
    except KeyError:
        deployment = None

    if deployment in DESK_HEIGHT_PRIOR_BY_DEPLOYMENT_M:
        return float(DESK_HEIGHT_PRIOR_BY_DEPLOYMENT_M[deployment])

    return max(0.0, float(np.min(nodes[:, 2]) - 2.0))


def get_geometry_embedding_features(
    layout: str,
    desk_id: int,
    tap_indices: list[float] | tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return invariant geometry features with shape [4 links, 5 features].

    Parameters
    ----------
    layout:
        Deployment string such as "deployment10", "Deployment10", or the full
        room name.
    desk_id:
        1-based desk id. Used for validation against the known deployment size.
    tap_indices:
        Four tap indices in this order: node0-1, node1-3, node2-3, node0-2.

    Returns
    -------
    np.ndarray
        Shape [4, 5], with columns:
        [baseline_m, excess_path_m, cos_beta, d_parallel_m, d_perp_m].

    Notes
    -----
    The returned excess_path_m column is the tap-derived measurement
    ``tap * 0.3002 / 2``. The desk XYZ used for the angle/parallel/perpendicular
    terms is estimated from the same four tap constraints.
    """
    deployment = resolve_deployment_key(layout)
    if not isinstance(desk_id, Integral):
        raise TypeError(f"desk_id must be an int, found {type(desk_id).__name__}")
    desk_id = int(desk_id)

    max_desk_id = len(estimated_tap_dict[deployment])
    if desk_id < 1 or desk_id > max_desk_id:
        raise ValueError(f"{deployment} has desk ids 1..{max_desk_id}, got {desk_id}")

    taps = np.asarray(tap_indices, dtype=float)
    if taps.shape != (len(TAP_PAIR_ORDER),):
        raise ValueError(f"tap_indices must have shape ({len(TAP_PAIR_ORDER)},), found {taps.shape}")

    nodes = standalone_node_xyz(deployment)
    desk_z_prior = DESK_HEIGHT_PRIOR_BY_DEPLOYMENT_M[deployment]
    return compute_geometry_features_from_node_taps(
        nodes,
        taps,
        desk_z_prior_m=desk_z_prior,
    )


def tap_targets_m(taps: list[float], nodes: np.ndarray) -> np.ndarray:
    if len(taps) != len(TAP_PAIR_ORDER):
        raise ValueError(f"Expected {len(TAP_PAIR_ORDER)} taps, got {len(taps)}")

    targets = []
    for tap, (i, j) in zip(taps, TAP_PAIR_ORDER):
        baseline = distance(nodes[i], nodes[j])
        targets.append(baseline + float(tap) * TAP_TO_EXCESS_LENGTH_M)
    return np.asarray(targets)


def solve_position_from_taps(
    nodes: np.ndarray,
    taps: list[float],
    initial: np.ndarray,
    bounds_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object]:
    targets = tap_targets_m(taps, nodes)

    def residuals(pos: np.ndarray) -> np.ndarray:
        return np.array(
            [
                distance(pos, nodes[i]) + distance(pos, nodes[j]) - target
                for target, (i, j) in zip(targets, TAP_PAIR_ORDER)
            ]
        )

    mins = np.min(bounds_points, axis=0)
    maxs = np.max(bounds_points, axis=0)
    lower = np.array([mins[0] - 1.0, mins[1] - 1.0, max(0.0, mins[2] - 0.5)])
    upper = np.array([maxs[0] + 1.0, maxs[1] + 1.0, maxs[2] + 0.5])

    center = np.mean(bounds_points, axis=0)
    starts = [
        initial,
        np.array([center[0], center[1], initial[2]]),
        np.array([nodes[:, 0].mean(), nodes[:, 1].mean(), initial[2]]),
    ]
    for x in np.linspace(mins[0], maxs[0], 3):
        for y in np.linspace(mins[1], maxs[1], 3):
            starts.append(np.array([x, y, initial[2]]))

    best = None
    for start in starts:
        start = np.clip(start, lower + 1e-6, upper - 1e-6)
        result = least_squares(
            residuals,
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.15,
            max_nfev=3000,
        )
        if best is None or result.cost < best.cost:
            best = result

    return best.x, residuals(best.x), best


def estimate_desk_xyz_from_node_taps(
    node_xyz: np.ndarray,
    tap_indices: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    desk_z_prior_m: float | None = None,
) -> np.ndarray:
    """Estimate desk XYZ from four bistatic tap indices and node coordinates."""
    nodes = coerce_node_xyz(node_xyz)
    taps = np.asarray(tap_indices, dtype=float)
    if taps.shape != (len(TAP_PAIR_ORDER),):
        raise ValueError(f"tap_indices must have shape ({len(TAP_PAIR_ORDER)},), found {taps.shape}")

    if desk_z_prior_m is None:
        desk_z_prior_m = max(0.0, float(np.min(nodes[:, 2]) - 2.0))

    xy_min = np.min(nodes[:, :2], axis=0)
    xy_max = np.max(nodes[:, :2], axis=0)
    xy_center = np.mean(nodes[:, :2], axis=0)
    bounds_points = np.vstack(
        [
            nodes,
            np.array([xy_min[0], xy_min[1], 0.0]),
            np.array([xy_max[0], xy_max[1], 0.0]),
            np.array([xy_center[0], xy_center[1], desk_z_prior_m]),
        ]
    )
    initial = np.array([xy_center[0], xy_center[1], desk_z_prior_m])
    desk_xyz, _, _ = solve_position_from_taps(nodes, taps.tolist(), initial, bounds_points)
    return desk_xyz


def compute_geometry_features_from_node_taps(
    node_xyz: np.ndarray,
    tap_indices: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    desk_z_prior_m: float | None = None,
) -> np.ndarray:
    """Return invariant geometry features with shape [4 links, 5 features]."""
    nodes = coerce_node_xyz(node_xyz)
    taps = np.asarray(tap_indices, dtype=float)
    if taps.shape != (len(TAP_PAIR_ORDER),):
        raise ValueError(f"tap_indices must have shape ({len(TAP_PAIR_ORDER)},), found {taps.shape}")

    desk_xyz = estimate_desk_xyz_from_node_taps(
        nodes,
        taps,
        desk_z_prior_m=desk_z_prior_m,
    )
    raw_geometry, _ = compute_symmetric_link_geometry(
        scene_xyz=desk_xyz[None, :],
        node_xyz=nodes,
        link_index=np.asarray(TAP_PAIR_ORDER, dtype=int),
    )
    features = raw_geometry[0]
    features[:, 1] = taps * TAP_TO_EXCESS_LENGTH_M
    return features


def generate_geometry_feature_lookup(
    node_coordinates_by_layout: dict | None = None,
    deployment_dict: dict | None = None,
    *,
    desk_height_prior_by_layout: dict | None = None,
    set_default: bool = True,
) -> dict[str, dict[int, np.ndarray]]:
    """Generate a nested lookup of invariant geometry features.

    Parameters
    ----------
    node_coordinates_by_layout:
        Mapping from layout/deployment name to node coordinates. Each value can
        be either a dict like {"node0": [x, y, z], ...} or a [4, 3] array.
        Defaults to the standalone optimized nodes in node_coordinate_dict.
    deployment_dict:
        Mapping from layout/deployment name to per-desk tap vectors. Example:
        {"deployment7": {"1": [19, 24, 42, 12], ...}}.
        Defaults to estimated_tap_dict.
    desk_height_prior_by_layout:
        Optional layout-to-desk-height prior used only to choose the vertical
        mirror solution when estimating desk XYZ from taps.
    set_default:
        If True, cache the returned lookup so query_geometry_feature_lookup can
        be called with only layout and desk_id.

    Returns
    -------
    dict[str, dict[int, np.ndarray]]
        lookup[normalized_layout][desk_id] -> [4, 5] feature array.
    """
    global DEFAULT_GEOMETRY_FEATURE_LOOKUP

    if node_coordinates_by_layout is None:
        node_coordinates_by_layout = node_coordinate_dict
    if deployment_dict is None:
        deployment_dict = estimated_tap_dict

    lookup: dict[str, dict[int, np.ndarray]] = {}
    for layout_key, taps_by_desk in deployment_dict.items():
        node_key = find_layout_mapping_key(node_coordinates_by_layout, str(layout_key))
        nodes = coerce_node_xyz(node_coordinates_by_layout[node_key])
        desk_z_prior = resolve_desk_height_prior(
            str(layout_key),
            nodes,
            desk_height_prior_by_layout=desk_height_prior_by_layout,
        )

        normalized_layout = normalize_layout_name(str(layout_key))
        lookup[normalized_layout] = {}
        for desk_id_raw, taps in taps_by_desk.items():
            desk_id = int(desk_id_raw)
            lookup[normalized_layout][desk_id] = compute_geometry_features_from_node_taps(
                nodes,
                taps,
                desk_z_prior_m=desk_z_prior,
            )

    if set_default:
        DEFAULT_GEOMETRY_FEATURE_LOOKUP = lookup
    return lookup


def query_geometry_feature_lookup(
    layout: str,
    desk_id: int,
    feature_lookup: dict[str, dict[int, np.ndarray]] | None = None,
) -> np.ndarray:
    """Query precomputed geometry features by layout and desk id.

    If feature_lookup is omitted, this uses a lazily generated default lookup
    built from node_coordinate_dict and estimated_tap_dict.
    """
    if feature_lookup is None:
        global DEFAULT_GEOMETRY_FEATURE_LOOKUP
        if DEFAULT_GEOMETRY_FEATURE_LOOKUP is None:
            DEFAULT_GEOMETRY_FEATURE_LOOKUP = generate_geometry_feature_lookup()
        feature_lookup = DEFAULT_GEOMETRY_FEATURE_LOOKUP

    layout_key = find_layout_mapping_key(feature_lookup, layout)
    desk_id = int(desk_id)
    if desk_id not in feature_lookup[layout_key]:
        available = sorted(feature_lookup[layout_key])
        raise KeyError(f"{layout!r} has desk ids {available}, got {desk_id}")
    return np.asarray(feature_lookup[layout_key][desk_id], dtype=float).copy()


def estimate_desks_from_taps(
    deployment: str,
    nodes: np.ndarray,
    optimized_desks: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    taps_by_desk = estimated_tap_dict[deployment]
    if optimized_desks:
        bounds_points = np.vstack([nodes, *optimized_desks.values()])
        default_z = float(np.mean([point[2] for point in optimized_desks.values()]))
    else:
        bounds_points = nodes
        default_z = max(0.0, float(np.min(nodes[:, 2]) - 2.0))

    results = {}
    for desk_key, taps in taps_by_desk.items():
        optimized = optimized_desks.get(desk_key)
        initial = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1]), default_z])

        estimated, residual_m, result = solve_position_from_taps(nodes, taps, initial, bounds_points)
        residual_tap = residual_m / TAP_TO_EXCESS_LENGTH_M
        results[desk_key] = {
            "taps": taps,
            "estimated": estimated,
            "optimized": optimized,
            "residual_m": residual_m,
            "residual_tap": residual_tap,
            "success": result.success,
            "cost": result.cost,
        }

    return results


def prepare_geometry_embedding(
    deployment: str,
    nodes: np.ndarray,
    results: dict[str, dict[str, object]],
    *,
    range_resolution_m: float,
    los_bin: int,
) -> GeometryEmbedding:
    desk_keys = list(results)
    link_index = np.asarray(TAP_PAIR_ORDER, dtype=int)
    scene_xyz = np.vstack([results[desk_key]["estimated"] for desk_key in desk_keys])

    optimized_scene_xyz = np.full_like(scene_xyz, np.nan)
    for desk_idx, desk_key in enumerate(desk_keys):
        optimized = results[desk_key]["optimized"]
        if optimized is not None:
            optimized_scene_xyz[desk_idx] = optimized

    tap_index = np.asarray([results[desk_key]["taps"] for desk_key in desk_keys], dtype=float)
    fitted_raw_geometry, fitted_excess_path_m = compute_symmetric_link_geometry(
        scene_xyz=scene_xyz,
        node_xyz=nodes,
        link_index=link_index,
    )

    excess_path_m = tap_index * TAP_TO_EXCESS_LENGTH_M

    # The tap is the observed CIR prior, so the embedding's rho channel uses
    # the tap-derived excess path. The geometric solve residual is still saved
    # separately as fitted_excess_path_m - excess_path_m.
    raw_geometry = fitted_raw_geometry.copy()
    raw_geometry[..., 1] = excess_path_m

    reference_bins = compute_reference_bins(
        excess_path_m,
        range_resolution_m=range_resolution_m,
        los_bin=los_bin,
    )

    return GeometryEmbedding(
        deployment=deployment,
        desk_names=[f"desk{desk_key}" for desk_key in desk_keys],
        link_index=link_index,
        scene_xyz=scene_xyz,
        optimized_scene_xyz=optimized_scene_xyz,
        tap_index=tap_index,
        raw_geometry=raw_geometry,
        fitted_raw_geometry=fitted_raw_geometry,
        excess_path_m=excess_path_m,
        fitted_excess_path_m=fitted_excess_path_m,
        reference_bins=reference_bins,
    )


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def report_embedding(deployment: str, csv_path: Path | None, results: dict[str, dict[str, object]]) -> None:
    print(f"\nDeployment: {deployment}")
    if csv_path:
        print(f"Loaded optimized coordinates: {csv_path}")
    else:
        print("Loaded fallback node coordinates from node_coordinate_dict")

    rows = []
    for desk_key, result in results.items():
        estimated = result["estimated"]
        optimized = result["optimized"]
        residual_tap = result["residual_tap"]
        tap_rmse = float(np.sqrt(np.mean(np.square(residual_tap))))
        if optimized is None:
            error = np.nan
            opt_cols = ["", "", ""]
        else:
            error = distance(estimated, optimized)
            opt_cols = [f"{optimized[0]:.3f}", f"{optimized[1]:.3f}", f"{optimized[2]:.3f}"]

        rows.append(
            [
                f"desk{desk_key}",
                f"{estimated[0]:.3f}",
                f"{estimated[1]:.3f}",
                f"{estimated[2]:.3f}",
                *opt_cols,
                "" if np.isnan(error) else f"{error:.3f}",
                f"{tap_rmse:.2f}",
            ]
        )

    print("\nTap-estimated desks vs optimized desks:")
    print_table(
        ["desk", "tap_x", "tap_y", "tap_z", "opt_x", "opt_y", "opt_z", "err_m", "tap_rmse"],
        rows,
    )


def report_geometry_embedding(embedding: GeometryEmbedding) -> None:
    residual_tap = (embedding.fitted_excess_path_m - embedding.excess_path_m) / TAP_TO_EXCESS_LENGTH_M
    print(
        "\nPrepared invariant geometry embedding: "
        f"raw_geometry shape {embedding.raw_geometry.shape} [desk, link, feature]"
    )
    print(f"Feature order: {', '.join(RAW_GEOMETRY_COLUMNS)}")
    print(f"Link order: {', '.join(f'{i}-{j}' for i, j in embedding.link_index)}")
    print(f"Reference bin range: {np.min(embedding.reference_bins):.2f} to {np.max(embedding.reference_bins):.2f}")
    print(f"Fitted-vs-tap excess residual RMSE: {np.sqrt(np.mean(residual_tap ** 2)):.3f} taps")


def save_embedding_csv(deployment: str, results: dict[str, dict[str, object]]) -> Path:
    path = OUTPUT_DIR / f"geometry_embedding_{deployment}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "desk",
            "tap_x_m",
            "tap_y_m",
            "tap_z_m",
            "optimized_x_m",
            "optimized_y_m",
            "optimized_z_m",
            "position_error_m",
            "tap_residual_rmse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for desk_key, result in results.items():
            estimated = result["estimated"]
            optimized = result["optimized"]
            residual_tap = result["residual_tap"]
            row = {
                "desk": f"desk{desk_key}",
                "tap_x_m": estimated[0],
                "tap_y_m": estimated[1],
                "tap_z_m": estimated[2],
                "tap_residual_rmse": float(np.sqrt(np.mean(np.square(residual_tap)))),
            }
            if optimized is not None:
                row.update(
                    {
                        "optimized_x_m": optimized[0],
                        "optimized_y_m": optimized[1],
                        "optimized_z_m": optimized[2],
                        "position_error_m": distance(estimated, optimized),
                    }
                )
            writer.writerow(row)
    return path


def save_geometry_features_csv(embedding: GeometryEmbedding) -> Path:
    path = OUTPUT_DIR / f"geometry_embedding_features_{embedding.deployment}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "deployment",
            "desk",
            "link",
            "tap_index",
            "reference_bin",
            *RAW_GEOMETRY_COLUMNS,
            "fitted_excess_path_m",
            "excess_residual_m",
            "excess_residual_tap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for desk_idx, desk_name in enumerate(embedding.desk_names):
            for link_idx, (node_a, node_b) in enumerate(embedding.link_index):
                fitted_excess = embedding.fitted_excess_path_m[desk_idx, link_idx]
                measured_excess = embedding.excess_path_m[desk_idx, link_idx]
                residual_m = fitted_excess - measured_excess
                row = {
                    "deployment": embedding.deployment,
                    "desk": desk_name,
                    "link": f"{node_a}-{node_b}",
                    "tap_index": embedding.tap_index[desk_idx, link_idx],
                    "reference_bin": embedding.reference_bins[desk_idx, link_idx],
                    "fitted_excess_path_m": fitted_excess,
                    "excess_residual_m": residual_m,
                    "excess_residual_tap": residual_m / TAP_TO_EXCESS_LENGTH_M,
                }
                for feature_idx, feature_name in enumerate(RAW_GEOMETRY_COLUMNS):
                    row[feature_name] = embedding.raw_geometry[desk_idx, link_idx, feature_idx]
                writer.writerow(row)
    return path


def save_geometry_features_npz(embedding: GeometryEmbedding) -> Path:
    path = OUTPUT_DIR / f"geometry_embedding_features_{embedding.deployment}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        deployment=np.asarray(embedding.deployment),
        desk_names=np.asarray(embedding.desk_names),
        link_index=embedding.link_index,
        raw_geometry=embedding.raw_geometry,
        raw_geometry_columns=np.asarray(RAW_GEOMETRY_COLUMNS),
        excess_path_m=embedding.excess_path_m,
        reference_bins=embedding.reference_bins,
        tap_index=embedding.tap_index,
        fitted_raw_geometry=embedding.fitted_raw_geometry,
        fitted_excess_path_m=embedding.fitted_excess_path_m,
        scene_xyz=embedding.scene_xyz,
        optimized_scene_xyz=embedding.optimized_scene_xyz,
    )
    return path


def set_plot_limits(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    x_lo, x_hi = mins[0] - PLOT_XY_MARGIN_M, maxs[0] + PLOT_XY_MARGIN_M
    y_lo, y_hi = mins[1] - PLOT_XY_MARGIN_M, maxs[1] + PLOT_XY_MARGIN_M
    z_lo, z_hi = max(0.0, mins[2] - PLOT_Z_MARGIN_M), maxs[2] + PLOT_Z_MARGIN_M

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    if FLIP_X_AXIS_IN_PLOT:
        ax.invert_xaxis()
    ax.set_box_aspect([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo])


def plot_embedding(
    deployment: str,
    nodes: np.ndarray,
    results: dict[str, dict[str, object]],
    show: bool,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / f"geometry_embedding_{deployment}.pdf"

    estimated_points = np.vstack([result["estimated"] for result in results.values()])
    optimized_points = [
        result["optimized"]
        for result in results.values()
        if result["optimized"] is not None
    ]
    all_points = [nodes, estimated_points]
    if optimized_points:
        all_points.append(np.vstack(optimized_points))

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=85, c="black", marker="s", label="Nodes")
    for idx, node in enumerate(nodes):
        ax.text(node[0], node[1], node[2] + 0.04, f"N{idx}", fontsize=10)

    for i, j in TAP_PAIR_ORDER:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            color="black",
            linewidth=1.0,
            alpha=0.28,
        )

    if optimized_points:
        optimized_array = np.vstack(optimized_points)
        ax.scatter(
            optimized_array[:, 0],
            optimized_array[:, 1],
            optimized_array[:, 2],
            s=95,
            c="dodgerblue",
            marker="^",
            edgecolors="black",
            label="Optimized desks",
            depthshade=False,
        )

    ax.scatter(
        estimated_points[:, 0],
        estimated_points[:, 1],
        estimated_points[:, 2],
        s=80,
        c="tab:red",
        marker="o",
        edgecolors="black",
        label="Tap-estimated desks",
        depthshade=False,
    )

    for desk_key, result in results.items():
        estimated = result["estimated"]
        optimized = result["optimized"]
        ax.text(estimated[0], estimated[1], estimated[2] + 0.04, f"D{desk_key}", fontsize=9, color="tab:red")
        if optimized is not None:
            ax.plot(
                [optimized[0], estimated[0]],
                [optimized[1], estimated[1]],
                [optimized[2], estimated[2]],
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
            )

    set_plot_limits(ax, np.vstack(all_points))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Geometry Embedding from Estimated Taps: {deployment}")
    ax.view_init(elev=24, azim=35)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight", pad_inches=0.20)
    if show:
        plt.show()
    plt.close(fig)
    return figure_path


def run_deployment(
    deployment_name: str,
    show: bool,
    no_plot: bool,
    no_csv: bool,
    no_npz: bool,
    range_resolution_m: float,
    los_bin: int,
) -> None:
    deployment, node_dict, optimized_desks, csv_path = load_optimized_geometry(deployment_name)
    nodes = nodes_as_array(node_dict)
    results = estimate_desks_from_taps(deployment, nodes, optimized_desks)
    embedding = prepare_geometry_embedding(
        deployment,
        nodes,
        results,
        range_resolution_m=range_resolution_m,
        los_bin=los_bin,
    )
    report_embedding(deployment, csv_path, results)
    report_geometry_embedding(embedding)

    if not no_csv:
        csv_out = save_embedding_csv(deployment, results)
        feature_csv_out = save_geometry_features_csv(embedding)
        print(f"Wrote embedding CSV: {csv_out}")
        print(f"Wrote geometry feature CSV: {feature_csv_out}")

    if not no_npz:
        feature_npz_out = save_geometry_features_npz(embedding)
        print(f"Wrote geometry feature NPZ: {feature_npz_out}")

    if not no_plot:
        figure_path = plot_embedding(deployment, nodes, results, show=show)
        print(f"Wrote embedding visualization: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed desk positions from estimated bistatic tap indices.")
    parser.add_argument(
        "--deployment",
        default="Deployment10",
        help="Deployment key or room name. Example: Deployment8 or Room_2_Deployment_8_Keysight.",
    )
    parser.add_argument("--all", action="store_true", help="Run all deployments in estimated_tap_dict.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV generation.")
    parser.add_argument("--no-npz", action="store_true", help="Skip NPZ geometry feature generation.")
    parser.add_argument(
        "--range-resolution-m",
        type=float,
        default=TAP_TO_EXCESS_LENGTH_M,
        help="Propagation excess-path length increment per CIR bin.",
    )
    parser.add_argument("--los-bin", type=int, default=0, help="LOS/reference bin index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deployments = list(estimated_tap_dict) if args.all else [args.deployment]
    for deployment in deployments:
        run_deployment(
            deployment,
            show=not args.no_show,
            no_plot=args.no_plot,
            no_csv=args.no_csv,
            no_npz=args.no_npz,
            range_resolution_m=args.range_resolution_m,
            los_bin=args.los_bin,
        )


if __name__ == "__main__":
    main()
