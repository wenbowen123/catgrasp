import time

from itertools import takewhile

from motion_planners.meta import direct_path
from .smoothing import smooth_path
from .rrt import TreeNode, configs
from .utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING, INF, elapsed_time, \
    negate


def asymmetric_extend(q1, q2, extend_fn, backward=False):
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)

def extend_towards(tree, target, distance_fn, extend_fn, collision_fn, swap, tree_frequency):
    last = argmin(lambda n: distance_fn(n.config, target), tree)
    extend = list(asymmetric_extend(last.config, target, extend_fn, swap))
    safe = list(takewhile(negate(collision_fn), extend))
    for i, q in enumerate(safe):
        if (i % tree_frequency == 0) or (i == len(safe) - 1):
            last = TreeNode(q, parent=last)
            tree.append(last)
    success = len(extend) == len(safe)
    return last, success

def rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn,
                iterations=RRT_ITERATIONS, tree_frequency=1, max_time=INF):
    start_time = time.time()
    assert tree_frequency >= 1
    nodes1, nodes2 = [TreeNode(q1)], [TreeNode(q2)]
    for iteration in irange(iterations):
        if max_time <= elapsed_time(start_time):
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        last1, _ = extend_towards(tree1, sample_fn(), distance_fn, extend_fn, collision_fn,
                                  swap, tree_frequency)
        last2, success = extend_towards(tree2, last1.config, distance_fn, extend_fn, collision_fn,
                                        not swap, tree_frequency)

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            #print('{} iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            return configs(path1[:-1] + path2[::-1])
    return None

#################################################################

def birrt(q1, q2, distance, sample, extend, collision, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING, max_time=INF,check_end_collision=True, **kwargs):
    '''
    @q1: start conf
    @q2: end conf
    @distance: dist func of two conf
    @collision: collision func
    '''
    start_time = time.time()
    # if collision(q1):
    #     print("[WARN] Birrt, start conf in collision")
    #     return None

    if check_end_collision:
        if collision(q2,debug=False):
            print("[WARN] Birrt, end conf in collision")
            return None

    path = direct_path(q1, q2, extend, collision,check_end_collision=check_end_collision)
    if path is not None:
        return path
    for attempt in irange(restarts + 1):
        if max_time <= elapsed_time(start_time):
            break
        path = rrt_connect(q1, q2, distance, sample, extend, collision,
                           max_time=max_time - elapsed_time(start_time), **kwargs)
        if path is not None:
            #print('{} attempts'.format(attempt))
            if smooth is None:
                return path
            return smooth_path(path, extend, collision, iterations=smooth)
    return None
