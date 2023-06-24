import numpy as np

from ors.preprocessing.ml.corner_points import corner_points
from ors.preprocessing.utils import (
    angle_of_quadrilateral,
    rotate_image_and_points,
    crop_quadrilateral_with_padding,
)


def extract_printjob_from_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:

    quadrilateral_points = corner_points(mask=mask)

    quadrilateral_points = np.array(quadrilateral_points) * [
        1 / mask.shape[1],
        1 / mask.shape[0],
    ]

    quadrilateral_points = np.array(quadrilateral_points) * [
        image.shape[1],
        image.shape[0],
    ]

    quadrilateral_points = quadrilateral_points.astype("int32").tolist()

    if quadrilateral_points is None:
        return None

    angle = angle_of_quadrilateral(quadrilateral_points)
    rotated_image, rotated_points = rotate_image_and_points(
        image, angle, quadrilateral_points
    )
    cropped_printjob = crop_quadrilateral_with_padding(rotated_image, rotated_points)

    return cropped_printjob