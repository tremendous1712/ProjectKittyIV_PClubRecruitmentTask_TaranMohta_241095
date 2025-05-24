import cv2
import numpy as np

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None, None
    Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return Px, Py

def cluster_points(points, nclusters=4):
    pts = np.float32(points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pts, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

def get_court_corners(image_path):
    """
    EXACTLY your original function:
    Convert to grayscale, threshold, Canny, HoughLinesP, filter, find intersections, cluster to 4, sort.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    v = np.median(thresh)
    lower = int(max(0, (1.0 - 0.33)*v))
    upper = int(min(255, (1.0 + 0.33)*v))
    edges = cv2.Canny(thresh, lower, upper, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
    if lines is None:
        raise ValueError("No lines detected")

    h_lines, v_lines = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if abs(y2 - y1) < abs(x2 - x1):
            h_lines.append((x1, y1, x2, y2))
        else:
            v_lines.append((x1, y1, x2, y2))

    filt_h, filt_v = [], []
    for l in h_lines:
        if all(abs(l[1] - o[1]) >= 10 for o in filt_h):
            filt_h.append(l)
    for l in v_lines:
        if all(abs(l[0] - o[0]) >= 10 for o in filt_v):
            filt_v.append(l)

    pts = []
    for h in filt_h:
        for v in filt_v:
            px, py = find_intersection(h, v)
            if px is not None:
                pts.append((px, py))

    if len(pts) < 4:
        raise ValueError("Not enough intersections detected for corners")

    centers = cluster_points(pts, 4)
    corners = sorted([(int(x), int(y)) for x, y in centers], key=lambda p: p[1])
    top = sorted(corners[:2], key=lambda p: p[0])
    bot = sorted(corners[2:], key=lambda p: p[0])
    # Order: [TL, TR, BR, BL]
    return [top[0], top[1], bot[1], bot[0]]
