"""
[ChatGPT generated code]
Module: convert_24_2_to_G_linear.py
Improvements over original script:
- clear function boundaries for unit testing
- dependency injection for get_coordinates to ease testing
- safer file handling and early folder creation
- consolidated plotting helper
- logging and input validation

This single file contains both the module and pytest unit tests at the bottom.
Save as two files if you prefer (module + tests) or run tests directly with pytest.

Usage (script mode):
    python convert_24_2_G_linear.py --input INPUT_DIR --output OUTPUT_DIR

Run tests:
    pip install pytest
    pytest convert_24_2_G_linear.py -q

"""

import os, sys
import argparse
import logging
from typing import Callable, Dict, Iterable, List, Sequence
custom_modules_path = os.path.expanduser('~/Google Drive/My Drive/Workspace/custom_modules/')
sys.path.append(custom_modules_path)
from module_data_manipulation import get_coordinates
import numpy as np
import matplotlib.pyplot as plt

# Default mapping (same as provided). Keep here but allow override in function calls for tests.
DEFAULT_MAP_G_FROM_24_2: Dict[int, List[int]] = {
    0: [15],
    1: [12],
    2: [38],
    3: [41],
    4: [23],
    5: [7],
    6: [17, 26],
    7: [32],
    8: [47],
    9: [35, 43],
    10: [31],
    11: [46],
    12: [28],
    13: [22],
    14: [6],
    15: [19],
    16: [23],
    17: [22],
    18: [31],
    19: [32],
    20: [2],
    21: [1],
    22: [51],
    23: [52],
    24: [20],
    25: [29],
    26: [26],
    27: [35],
    28: [18],
    29: [27],
    30: [3, 9, 17],
    31: [43, 49, 53],
    32: [22, 23, 31, 32],
    33: [24],
    34: [21],
    35: [30],
    36: [33],
    37: [3, 9],
    38: [0, 4],
    39: [44, 50],
    40: [49, 53],
    41: [9, 17],
    42: [4, 10],
    43: [36, 44],
    44: [43, 49],
    45: [14],
    46: [13],
    47: [39],
    48: [40],
    49: [8, 9, 15, 16],
    50: [4, 5, 11, 12],
    51: [37, 38, 44, 45],
    52: [41, 42, 48, 49],
    53: [0, 4, 10],
    54: [36, 44, 50],
    55: [2, 3],
    56: [0, 1],
    57: [50, 51],
    58: [52, 53]
}


# ------------------------- Core functions -------------------------

def inverse_distance_weights(xy_point: np.ndarray, xy_neighs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute inverse-distance weights for neighbors of a point.

    Parameters
    - xy_point: shape (2,) float array
    - xy_neighs: shape (N,2) float array
    - eps: small value to avoid division by zero (if a distance is zero we use eps)

    Returns
    - weights: shape (N,) that sums to 1. If xy_neighs is empty returns empty array.
    """
    xy_point = np.asarray(xy_point, dtype=float)
    xy_neighs = np.asarray(xy_neighs, dtype=float)

    if xy_neighs.size == 0:
        return np.array([])

    dists = np.linalg.norm(xy_neighs - xy_point.reshape(1, 2), axis=1)
    # replace zeros with eps so weight becomes very large for exact matches
    dists = np.where(dists <= 0.0, eps, dists)
    inv = 1.0 / dists
    weights = inv / inv.sum()
    return weights


def compute_weighted_average(xy_point: np.ndarray, values_neighbors: Sequence[float], xy_neighbors: np.ndarray) -> float:
    """Compute weighted average value for a target point given neighbor coordinates and their values.

    If there is a single neighbor, return that neighbor's value. If neighbor list is empty, raise ValueError.
    """
    values_neighbors = np.asarray(values_neighbors, dtype=float)
    xy_neighbors = np.asarray(xy_neighbors, dtype=float)

    if values_neighbors.size == 0:
        raise ValueError("values_neighbors must contain at least one value")

    if xy_neighbors.shape[0] != values_neighbors.size:
        raise ValueError("xy_neighbors and values_neighbors must have the same length")

    if values_neighbors.size == 1:
        return float(values_neighbors[0])

    weights = inverse_distance_weights(xy_point, xy_neighbors)
    return float(np.sum(values_neighbors * weights))


def process_file(input_path: str,
                 output_path: str,
                 xy_24_2: np.ndarray,
                 xy_G: np.ndarray,
                 map_G_from_24_2: Dict[int, List[int]] = DEFAULT_MAP_G_FROM_24_2) -> None:
    """Read one 24-2 text file and write out a mapped G pattern file and PNG plots.

    input file should contain a single line with comma-separated numeric values matching len(xy_24_2).
    """
    with open(input_path, 'r') as f:
        line = f.read().strip()
        if not line:
            raise ValueError(f"Empty input file: {input_path}")
        values_24_2 = np.array([float(x) for x in line.split(',')], dtype=float)

    if values_24_2.size != xy_24_2.shape[0]:
        raise ValueError(f"Input values length ({values_24_2.size}) does not match xy_24_2 ({xy_24_2.shape[0]})")

    num_locs_G = xy_G.shape[0]
    values_G = np.zeros(num_locs_G, dtype=float)

    for target_idx, source_idxs in map_G_from_24_2.items():
        if any(i < 0 or i >= values_24_2.size for i in source_idxs):
            raise IndexError(f"Mapping contains invalid source index for target {target_idx}: {source_idxs}")
        neigh_vals = values_24_2[source_idxs]
        neigh_xy = xy_24_2[source_idxs]
        values_G[target_idx] = compute_weighted_average(xy_G[target_idx], neigh_vals, neigh_xy)

    # save numeric result
    with open(output_path, 'w') as f:
        f.write(','.join(f"{v:.4f}" for v in values_G))

    # save visualization (optional, but kept to replicate original behavior)
    base = os.path.splitext(output_path)[0]
    _plot_overlap(xy_24_2, values_24_2, xy_G, values_G, f"{base}_overlap.png")
    _plot_points(xy_24_2, values_24_2, f"{base}_24_2.png")
    _plot_points(xy_G, values_G, f"{base}_G.png", color='red')


# ------------------------- Plot helpers -------------------------

def _plot_points(xy: np.ndarray, values: np.ndarray, filename: str, color: str = 'black') -> None:
    plt.figure()
    plt.xlim([-30, 30])
    plt.ylim([-30, 30])
    for k in range(xy.shape[0]):
        x, y = xy[k]
        plt.text(x, y, f"{values[k]:.1f}")
    plt.plot([0, 0], [-30, 30], 'r')
    plt.plot([-30, 30], [0, 0], 'r')
    plt.title(os.path.basename(filename))
    plt.savefig(filename)
    plt.close()


def _plot_overlap(xy1: np.ndarray, vals1: np.ndarray, xy2: np.ndarray, vals2: np.ndarray, filename: str) -> None:
    plt.figure()
    plt.xlim([-30, 30])
    plt.ylim([-30, 30])
    for k in range(xy1.shape[0]):
        x, y = xy1[k]
        plt.text(x, y, f"{vals1[k]:.1f}")
    for k in range(xy2.shape[0]):
        x, y = xy2[k]
        plt.text(x, y, f"{vals2[k]:.1f}", color='red')
    plt.plot([0, 0], [-30, 30], 'r')
    plt.plot([-30, 30], [0, 0], 'r')
    plt.title(os.path.basename(filename))
    plt.savefig(filename)
    plt.close()


# ------------------------- High-level processing -------------------------

def process_all_files(input_folder: str,
                      output_folder: str,
                      xy_24_2: np.ndarray,
                      xy_G: np.ndarray,
                      map_G_from_24_2: Dict[int, List[int]] = DEFAULT_MAP_G_FROM_24_2,
                      pattern: str = 'norm_242_{}.txt',
                      indices: Iterable[int] = range(1, 100)) -> List[str]:
    """Process a set of files and return the list of written output paths.

    This function is easy to unit test by using temporary directories and synthetic get_coordinates.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

    written_files = []
    for i in indices:
        input_filename = os.path.join(input_folder, pattern.format(i))
        output_filename = os.path.join(output_folder, f"norm_G_{i}.txt")
        if not os.path.exists(input_filename):
            logging.warning(f"Missing input file: {input_filename}")
            continue
        process_file(input_filename, output_filename, xy_24_2, xy_G, map_G_from_24_2)
        written_files.append(output_filename)
    return written_files


# ------------------------- CLI -------------------------

def default_get_coordinates(name: str, laterality: str = 'right') -> np.ndarray:
    """Thin wrapper to call external module — can be monkeypatched in tests."""
    from module_data_manipulation import get_coordinates

    return get_coordinates(name, laterality=laterality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    xy_24_2 = default_get_coordinates('24-2', laterality='right')
    xy_G = default_get_coordinates('G', laterality='right')

    process_all_files(args.input, args.output, xy_24_2, xy_G)


# ------------------------- Unit tests (pytest) -------------------------

# To run tests in this file run: pytest map_24_2_to_G_with_tests.py -q


def _make_square_coords(n_points: int) -> np.ndarray:
    """Return n_points coordinates placed on a grid for testing (deterministic)."""
    rng = np.random.default_rng(0)
    return rng.uniform(-10, 10, size=(n_points, 2))


def test_inverse_distance_weights_single_and_zero():
    p = np.array([0.0, 0.0])
    neighs = np.array([[0.0, 0.0], [1.0, 0.0]])
    w = inverse_distance_weights(p, neighs, eps=1e-6)
    # first neighbor is identical, so it should get the overwhelming weight
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > 0.999


def test_compute_weighted_average_trivial():
    p = np.array([0.0, 0.0])
    neighs = np.array([[0.0, 0.0]])
    vals = [42.0]
    assert compute_weighted_average(p, vals, neighs) == 42.0


def test_compute_weighted_average_two_points():
    p = np.array([0.0, 0.0])
    neighs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    vals = [10.0, 20.0]
    # symmetric distances; weights equal -> average
    out = compute_weighted_average(p, vals, neighs)
    assert np.isclose(out, 15.0)

# The below test added by Serife
def test_compute_weighted_average_three_points():
    p = np.array([2.0, 3.0])
    neighs = np.array([[3.0, 3.0], [5.0, 2.0], [10.0, 4.0]])
    vals = [0, 3, 7]
    out = compute_weighted_average(p, vals, neighs)
    print(out)
    assert np.isclose(out, 1.261525)


def test_process_all_files_temp(tmp_path, monkeypatch):
    # create synthetic coordinates and mapping for a small example
    xy_24_2 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    xy_G = np.array([[1.0, 1.0], [9.0, 1.0]])
    small_map = {0: [0, 1], 1: [1, 2]}

    # write two input files (only index 1 and 2 are used)
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    # file 1
    (in_dir / "norm_242_1.txt").write_text("1.0,2.0,3.0")
    # file 2
    (in_dir / "norm_242_2.txt").write_text("2.0,4.0,6.0")

    # run processing
    written = process_all_files(str(in_dir), str(out_dir), xy_24_2, xy_G, map_G_from_24_2=small_map, indices=[1, 2])

    # check outputs exist
    assert len(written) == 2
    for p in written:
        assert os.path.exists(p)
        content = open(p, 'r').read().strip()
        parts = [float(x) for x in content.split(',')]
        assert len(parts) == xy_G.shape[0]


if __name__ == '__main__':
    # if executed as script, run main()
    main()
