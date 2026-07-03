import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("./models/random_forest.pkl")

# 9 classes in alphabetical order (scikit-learn sorts class labels alphabetically)
# Format: all_cpu_features_{cpu_load}_memory_features_{mem_load}
# Badness weight: 0.0 = best node, 1.0 = worst node
# CPU weight contributes 60%, memory weight 40% (resource-congestion bias)
CLASS_LABELS = [
    "cpu_high_mem_high",    # 0
    "cpu_high_mem_medium",  # 1
    "cpu_high_mem_no",      # 2
    "cpu_medium_mem_high",  # 3
    "cpu_medium_mem_medium",# 4
    "cpu_medium_mem_no",    # 5
    "cpu_no_mem_high",      # 6
    "cpu_no_mem_medium",    # 7
    "cpu_no_mem_no",        # 8
]

#              cpu_load   mem_load   cpu_weight(0.6) + mem_weight(0.4)
CLASS_WEIGHTS = np.array([
    1.0 * 0.6 + 1.0 * 0.4,   # 0: cpu_high   + mem_high    = 1.00
    1.0 * 0.6 + 0.5 * 0.4,   # 1: cpu_high   + mem_medium  = 0.80
    1.0 * 0.6 + 0.0 * 0.4,   # 2: cpu_high   + mem_no      = 0.60
    0.5 * 0.6 + 1.0 * 0.4,   # 3: cpu_medium + mem_high    = 0.70
    0.5 * 0.6 + 0.5 * 0.4,   # 4: cpu_medium + mem_medium  = 0.50
    0.5 * 0.6 + 0.0 * 0.4,   # 5: cpu_medium + mem_no      = 0.30
    0.0 * 0.6 + 1.0 * 0.4,   # 6: cpu_no     + mem_high    = 0.40
    0.0 * 0.6 + 0.5 * 0.4,   # 7: cpu_no     + mem_medium  = 0.20
    0.0 * 0.6 + 0.0 * 0.4,   # 8: cpu_no     + mem_no      = 0.00 (best)
])


class RFModel:
    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        self._model = joblib.load(path)

        # verify class count matches expectation
        n_classes = len(self._model.classes_)
        if n_classes != 9:
            raise ValueError(
                f"Expected 9 classes, model has {n_classes}. "
                f"Actual classes: {self._model.classes_}"
            )

        print(f"[model] loaded from {path}")
        print(f"[model] classes ({n_classes}):")
        for i, (label, weight) in enumerate(zip(self._model.classes_, CLASS_WEIGHTS)):
            print(f"  [{i}] {label}  →  badness={weight:.2f}")

    def predict(self, feature_matrix: np.ndarray, node_names: list[str]) -> dict[str, float]:
        """
        feature_matrix: shape (n_nodes, 38)
        node_names: list of node names matching row order in feature_matrix
        returns: dict mapping node_name -> scheduler score (0.0-1.0, higher = less congested)
        """
        if feature_matrix.shape[0] == 0:
            return {}

        if feature_matrix.shape[1] != 38:
            raise ValueError(
                f"Expected 38 features, got {feature_matrix.shape[1]}"
            )

        proba = self._model.predict_proba(feature_matrix)  # (n_nodes, 9)

        # weighted average congestion score: dot product with badness weights
        congestion = proba @ CLASS_WEIGHTS  # (n_nodes,), range 0.0-1.0

        # invert: low congestion → high scheduler score (1.0 = best node)
        scheduler_scores = 1.0 - congestion

        # also log the predicted class per node for observability
        predicted_indices = np.argmax(proba, axis=1)
        for node, idx,con, score in zip(node_names, predicted_indices, congestion, scheduler_scores):
            print(
                f"[model] {node}: class={CLASS_LABELS[idx]} "
                f"congestion={con} "
                f"scheduler_score={score:.3f}"
            )

        return {
            node: round(float(score), 4)
            for node, score in zip(node_names, scheduler_scores)
        }


# singleton
model = RFModel()