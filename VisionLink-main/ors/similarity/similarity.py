from typing import Dict, Iterable, Tuple

import numpy as np


def calc_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.sqrt(np.sum((vector1 - vector2) ** 2))


def find_best_match(
    feature_vector: np.ndarray, feature_vector_set: Dict[int, np.ndarray]
) -> Tuple[int, float]:

    shortest_distance = float("inf")
    best_match = 0
    for jobnumber, features in feature_vector_set.items():
        distance = calc_distance(feature_vector, features)
        if distance < shortest_distance:
            shortest_distance = distance
            best_match = jobnumber

    return best_match, shortest_distance
