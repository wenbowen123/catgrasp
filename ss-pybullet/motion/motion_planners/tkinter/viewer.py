try:
    from Tkinter import Tk, Canvas, Toplevel, LAST
    #import TKinter as tk
except ModuleNotFoundError:
    from tkinter import Tk, Canvas, Toplevel, LAST
    #import tkinter as tk

import numpy as np

from collections import namedtuple

from motion_planners.utils import pairs, get_delta

Box = namedtuple('Box', ['lower', 'upper'])
Circle = namedtuple('Circle', ['center', 'radius'])

class PRMViewer(object):
    def __init__(self, width=500, height=500, title='PRM', background='tan'):
        tk = Tk()
        tk.withdraw()
        top = Toplevel(tk)
        top.wm_title(title)
        top.protocol('WM_DELETE_WINDOW', top.destroy)
        self.width = width
        self.height = height
        self.canvas = Canvas(top, width=self.width, height=self.height, background=background)
        self.canvas.pack()

    def pixel_from_point(self, point):
        (x, y) = point
        # return (int(x*self.width), int(self.height - y*self.height))
        return (x * self.width, self.height - y * self.height)

    def draw_point(self, point, radius=5, color='black'):
        (x, y) = self.pixel_from_point(point)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline='')

    def draw_line(self, segment, color='black'):
        (point1, point2) = segment
        (x1, y1) = self.pixel_from_point(point1)
        (x2, y2) = self.pixel_from_point(point2)
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1)

    def draw_arrow(self, point1, point2, color='black'):
        (x1, y1) = self.pixel_from_point(point1)
        (x2, y2) = self.pixel_from_point(point2)
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, arrow=LAST)

    def draw_rectangle(self, box, width=2, color='brown'):
        (point1, point2) = box
        (x1, y1) = self.pixel_from_point(point1)
        (x2, y2) = self.pixel_from_point(point2)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, width=width)

    def draw_circle(self, center, radius, width=2, color='black'):
        (x1, y1) = self.pixel_from_point(np.array(center) - radius * np.ones(2))
        (x2, y2) = self.pixel_from_point(np.array(center) + radius * np.ones(2))
        self.canvas.create_oval(x1, y1, x2, y2, outline='black', fill=color, width=width)

    def clear(self):
        self.canvas.delete('all')


#################################################################

def contains(q, box):
    (lower, upper) = box
    return np.greater_equal(q, lower).all() and \
           np.greater_equal(upper, q).all()
    #return np.all(q >= lower) and np.all(upper >= q)

def point_collides(point, boxes):
    return any(contains(point, box) for box in boxes)

def sample_line(segment, step_size=.02):
    (q1, q2) = segment
    diff = get_delta(q1, q2)
    dist = np.linalg.norm(diff)
    for l in np.arange(0., dist, step_size):
        yield tuple(np.array(q1) + l * diff / dist)
    yield q2

def line_collides(line, box):  # TODO - could also compute this exactly
    return any(point_collides(point, boxes=[box]) for point in sample_line(line))

def is_collision_free(line, boxes):
    return not any(line_collides(line, box) for box in boxes)

def create_box(center, extents):
    (x, y) = center
    (w, h) = extents
    lower = (x - w / 2., y - h / 2.)
    upper = (x + w / 2., y + h / 2.)
    return Box(np.array(lower), np.array(upper))

def get_box_center(box):
    lower, upper = box
    return np.average([lower, upper], axis=0)

def get_box_extent(box):
    lower, upper = box
    return get_delta(lower, upper)

def sample_box(box):
    (lower, upper) = box
    return np.random.random(len(lower)) * get_box_extent(box) + lower

#################################################################

def draw_environment(obstacles, regions):
    viewer = PRMViewer()
    for box in obstacles:
        viewer.draw_rectangle(box, color='brown')
    for name, region in regions.items():
        if name != 'env':
            viewer.draw_rectangle(region, color='green')
    return viewer

def add_segments(viewer, segments, **kwargs):
    if segments is None:
        return
    for line in segments:
        viewer.draw_line(line, **kwargs)
        #for p in [p1, p2]:
        for p in sample_line(line):
            viewer.draw_point(p, radius=2, **kwargs)

def add_path(viewer, path, **kwargs):
    segments = list(pairs(path))
    return add_segments(viewer, segments, **kwargs)

def draw_solution(segments, obstacles, regions):
    viewer = draw_environment(obstacles, regions)
    add_segments(viewer, segments)

def add_roadmap(viewer, roadmap, **kwargs):
    for line in roadmap:
        viewer.draw_line(line, **kwargs)

def draw_roadmap(roadmap, obstacles, regions):
    viewer = draw_environment(obstacles, regions)
    add_roadmap(viewer, roadmap)

def add_points(viewer, points, **kwargs):
    for sample in points:
        viewer.draw_point(sample, **kwargs)

def get_distance_fn(weights):
    difference_fn = get_delta
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn
