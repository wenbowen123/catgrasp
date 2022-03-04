from random import randint, random
from motion_planners.utils import INF, elapsed_time, irange, waypoints_from_path, pairs, get_distance, \
    convex_combination, flatten

import time
import numpy as np

def smooth_path_old(path, extend, collision, iterations=50, max_tine=INF):
    start_time = time.time()
    smoothed_path = path
    for _ in irange(iterations):
        if elapsed_time(start_time) > max_tine:
            break
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path

def smooth_path(path, extend, collision, distance_fn=None, iterations=50, max_tine=INF):
    # TODO: makes an assumption on the distance metric
    # TODO: smooth until convergence
    start_time = time.time()
    if distance_fn is None:
        distance_fn = get_distance
    waypoints = path
    for _ in irange(iterations):
        waypoints = waypoints_from_path(waypoints)
        if (elapsed_time(start_time) > max_tine) or (len(waypoints) <= 2):
            break

        indices = list(range(len(waypoints)))
        segments = list(pairs(indices))
        distances = [distance_fn(waypoints[i], waypoints[j]) for i, j in segments]
        probabilities = np.array(distances) / sum(distances)

        #segment1, segment2 = choices(segments, weights=probabilities, k=2)
        seg_indices = list(range(len(segments)))
        seg_idx1, seg_idx2 = np.random.choice(seg_indices, size=2, replace=True, p=probabilities)
        segment1, segment2 = segments[seg_idx1], segments[seg_idx2]
        if segment1 == segment2: # choices samples with replacement
            continue
        if segment2[1] <= segment1[0]:
            segment1, segment2 = segment2, segment1
        point1, point2 = [convex_combination(waypoints[i], waypoints[j], w=random())
                          for i, j in [segment1, segment2]]
        if all(not collision(q) for q in extend(point1, point2)):
            i, _ = segment1
            _, j = segment2
            waypoints = waypoints[:i + 1] + [point1, point2] + waypoints[j:]
    return list(flatten(extend(q1, q2) for q1, q2 in pairs(waypoints)))
