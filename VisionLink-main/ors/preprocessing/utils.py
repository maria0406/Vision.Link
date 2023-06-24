import math

import cv2
import numpy as np


def angle_of_quadrilateral(points):
    points.sort(key=lambda x: x[1])  # sort points based on y-coordinate
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Calculate the angle of the line
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle = math.atan2(delta_y, delta_x)
    angle = math.degrees(angle)

    return angle


def rotate_image_and_points(image, angle, points):
    # Get the size of the image
    rows, cols = image.shape[:2]

    # Calculate the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Get the size of the bounding box surrounding the rotated image
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_cols = int((rows * sin) + (cols * cos))
    new_rows = int((rows * cos) + (cols * sin))

    # Adjust the rotation matrix to take into account the translation
    rot_mat[0, 2] += (new_cols / 2) - (cols / 2)
    rot_mat[1, 2] += (new_rows / 2) - (rows / 2)

    # Rotate the image
    rotated_image = cv2.warpAffine(
        image,
        rot_mat,
        (new_cols, new_rows),
        flags=cv2.INTER_CUBIC,
        borderValue=(255, 255, 255),
    )

    # Rotate the points
    rotated_points = []
    for point in points:
        x, y = point
        rot_point = np.dot(rot_mat, np.array([x, y, 1]))
        rotated_points.append(rot_point[:2].astype(int))

    return rotated_image, rotated_points


def crop_quadrilateral_with_padding(image, points, padding=1):
    # Create a list of the four points as integers
    points = [tuple(map(int, point)) for point in points]

    # Find the top-left and bottom-right points of the bounding rectangle
    x_min = min([point[0] for point in points])
    x_max = max([point[0] for point in points])
    y_min = min([point[1] for point in points])
    y_max = max([point[1] for point in points])

    # Add padding to the bounding rectangle
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Ensure that the bounding rectangle stays within the image
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1] - 1)
    y_max = min(y_max, image.shape[0] - 1)

    # Crop the image
    crop = image[y_min:y_max, x_min:x_max]

    return crop