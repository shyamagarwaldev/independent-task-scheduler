import numpy as np

NETWORK_FEATURES = [
    "rx_drop",
    "tx_drop",
    "rx_bytes",
    "tx_bytes",
    "retSeg",
]

#               rx_drop tx_drop rx_bytes tx_bytes retSeg
WEIGHTS = np.array([0.25, 0.25, 0.20, 0.20, 0.10])


def network_score(
    node_names: list[str],
    network_matrix: np.ndarray,
) -> dict[str, float]:
    """
    network_matrix shape = (n_nodes, 5)

    Returns:
        {
            "worker-1": 0.92,
            "worker-2": 0.41,
            ...
        }
    """

    if network_matrix.shape[1] != len(NETWORK_FEATURES):
        raise ValueError("Expected 5 network features")

    normalized = np.zeros_like(network_matrix, dtype=np.float64)

    for col in range(network_matrix.shape[1]):
        values = network_matrix[:, col]

        mn = values.min()
        mx = values.max()

        if np.isclose(mx, mn):
            normalized[:, col] = 0.0
        else:
            normalized[:, col] = (values - mn) / (mx - mn)

    # Higher values mean more congestion, so invert
    scores = 1.0 - (normalized @ WEIGHTS)

    return {
        node: round(float(score), 4)
        for node, score in zip(node_names, scores)
    }