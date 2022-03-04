from itertools import combinations, product, permutations
from scipy.spatial.kdtree import KDTree

import numpy as np

from motion_planners.utils import INF, compute_path_cost, get_distance

def compute_median_distance(path1, path2):
    differences = [get_distance(q1, q2) for q1, q2 in product(path1, path2)]
    return np.median(differences)


def compute_minimax_distance(path1, path2):
    overall_distance = 0.
    for path, other in permutations([path1, path2]):
        tree = KDTree(other)
        for q1 in path:
            #closest_distance = min(get_distance(q1, q2) for q2 in other)
            closest_distance, closest_index = tree.query(q1, k=1, eps=0.)
            overall_distance = max(overall_distance, closest_distance)
    return overall_distance


def compute_portfolio_distance(path1, path2, min_distance=0.):
    # TODO: generic distance_fn
    # TODO: min_distance from stats about the portfolio
    distance = compute_minimax_distance(path1, path2)
    if distance < min_distance:
        return 0.
    return sum(compute_path_cost(path, get_distance) for path in [path1, path2])


def score_portfolio(portfolio, **kwargs):
    # TODO: score based on collision voxel overlap at different resolutions
    score_fn = compute_minimax_distance # compute_median_distance | compute_minimax_distance | compute_portfolio_distance
    score = INF
    for path1, path2 in combinations(portfolio, r=2):
        score = min(score, score_fn(path1, path2, **kwargs))
    return score


def exhaustively_select_portfolio(candidates, k=10, **kwargs):
    if len(candidates) <= k:
        return candidates
    # TODO: minimum length portfolio such that at nothing is within a certain distance
    best_portfolios, best_score = [], 0
    for portfolio in combinations(candidates, r=k):
        score = score_portfolio(portfolio, **kwargs)
        if score > best_score:
            best_portfolios, best_score = portfolio, score
    return best_portfolios

def greedily_select_portfolio(candidates, k=10):
    # Higher score is better
    if len(candidates) <= k:
        return candidates
    raise NotImplementedError()
    #return best_portfolios
