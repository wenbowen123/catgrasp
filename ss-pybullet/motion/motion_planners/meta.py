import time

from motion_planners.smoothing import smooth_path
from motion_planners.utils import RRT_RESTARTS, RRT_SMOOTHING, INF, irange, elapsed_time, compute_path_cost


def direct_path(q1, q2, extend_fn, collision_fn, check_end_collision=True):
    # TODO: version which checks whether the segment is valid
    if collision_fn(q1):
        return None
    if check_end_collision:
        if collision_fn(q2):
            return None

    path = [q1]
    for q in extend_fn(q1, q2):
        if collision_fn(q):
            return None
        path.append(q)
    return path


def random_restarts(solve_fn, q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                    restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING,
                    success_cost=0., max_time=INF, max_solutions=1, **kwargs):
    start_time = time.time()
    solutions = []
    if any(collision_fn(q) for q in [q1, q2]):
        return solutions
    path = direct_path(q1, q2, extend_fn, collision_fn)
    if path is not None:
        solutions.append(path)

    for attempt in irange(restarts + 1):
        if (len(solutions) >= max_solutions) or (elapsed_time(start_time) > max_time):
            break
        attempt_time = (max_time - elapsed_time(start_time))
        path = solve_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                        max_time=attempt_time, **kwargs)
        if path is None:
            continue
        if smooth is not None:
            path = smooth_path(path, extend_fn, collision_fn, iterations=smooth)
        solutions.append(path)
        if compute_path_cost(path, distance_fn) < success_cost:
            break
    solutions = sorted(solutions, key=lambda path: compute_path_cost(path, distance_fn))
    print('Solutions ({}): {} | Time: {:.3f}'.format(len(solutions), [(len(path), round(compute_path_cost(
        path, distance_fn), 3)) for path in solutions], elapsed_time(start_time)))

    return solutions