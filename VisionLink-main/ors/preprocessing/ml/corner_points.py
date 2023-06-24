import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon


def corner_points(mask):

    contours = find_contours(mask, 0.5)
    polygon = Polygon(contours[0])

    # Get the minimum bounding box of the object
    min_x, min_y, max_x, max_y = polygon.minimum_rotated_rectangle.bounds
    min_rect_pts = np.array(polygon.minimum_rotated_rectangle.exterior.coords[:-1], dtype=np.int32)

    return [[int(x), int(y)] for y, x in min_rect_pts]