#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Point of Interest module 
# Image Pipeline Processing 4 Astronomy
# Copyrights Nicolas Brunie (2016-), nibrunie@gmail.com
# https://github.com/nibrunie/ipp4a

import math

def exp_weight(v0, v1):
  return math.exp(abs(v0[2] - v1[2]))

## compute weighted distance between two points
def distance(p0, p1, weight = (lambda v0, v1: 1.0)):
  dx = (p0[0] - p1[0]) **2
  dy = (p0[1] - p1[1]) **2
  return math.sqrt(dx + dy) * weight(p0, p1)

# erase points that are too close, as they may be duplicate
def delete2Close(point_list, threshold = 5.0):
  result = []
  # computing distance matrix
  for i in xrange(len(point_list)):
    add = True
    for j in xrange(len(point_list)):
      if i != j and distance(point_list[i], point_list[j], exp_weight) < threshold:
        add = False
        break
    if add: 
      result.append(point_list[i])
  return result
      


# compute the list of pairs of closest points from 
# list0 and list1. Only includes points that are
# closer than threshold
def compute_pairs(list0, list1, threshold = 10.0):
  result = []
  valid_list1 = [i for i in xrange(len(list1))]
  for p0 in list0:
    ordered_list = [(i, distance(p0, list1[i])) for i in valid_list1]
    ordered_list.sort(key = lambda v: v[1])
    elected, d = ordered_list[0]
    if d < threshold: result.append((p0, list1[elected]))
    valid_list1.remove(elected)
  return result


class Transform:
  def __init__(self, b, A, db, dA):
    self.b = b
    self.A = A
    self.db = db
    self.dA = dA

  def apply(self, v):
    return (self.b[0] + self.A[0][0] * v[0] + self.A[0][1] * v[1], 
            self.b[1] + self.A[1][0] * v[0] + self.A[1][1] * v[1],
            v[2])

## Perform an iteration of Iterative Closest Point algorithm
def icp_iteration(ref_list, moved_list, tfm):
  closest_pairs = compute_pairs(ref_list, moved_list)

  dG_dbx = sum((2  *  (p0[0] - p1[0])) for (p0, p1) in closest_pairs)
  dG_dby = sum((2  *  (p0[1] - p1[1])) for (p0, p1) in closest_pairs)
  dG_dA00 = sum(2  *  p0[0] * (p0[0] - p1[0]) for (p0, p1) in closest_pairs)
  dG_dA01 = sum(2  *  p0[1] * (p0[0] - p1[0]) for (p0, p1) in closest_pairs)
  dG_dA10 = sum(2  *  p0[0] * (p0[1] - p1[1]) for (p0, p1) in closest_pairs)
  dG_dA11 = sum(2  *  p0[1] * (p0[1] - p1[1]) for (p0, p1) in closest_pairs)

  # goal is to minimize G so to update in the direction
  # it decreases <=> dG_dV is negative
  dbx = tfm.db if dG_dbx > 0 else -tfm.db
  dby = tfm.db if dG_dby > 0 else -tfm.db

  dA00 = tfm.dA if dG_dA00 > 0 else -tfm.dA
  dA01 = tfm.dA if dG_dA01 > 0 else -tfm.dA
  dA10 = tfm.dA if dG_dA10 > 0 else -tfm.dA
  dA11 = tfm.dA if dG_dA11 > 0 else -tfm.dA

  tfm.b[0]    += dbx 
  tfm.b[1]    += dby 
  tfm.A[0][0] += dA00
  tfm.A[0][1] += dA01
  tfm.A[1][0] += dA10
  tfm.A[1][1] += dA11
  return tfm

def extract_weight(point):
  return point[2]

def exclude_sinkhole_neihbourg(point_list, sinkhole_minweight = 100.0, min_distance = lambda v: 2 * math.sqrt(extract_weight(v)) / math.pi):
  sinkhole_list = [point for point in point_list if extract_weight(point) >= sinkhole_minweight]
  result = []
  for point in point_list:
    valid = True
    for sinkhole in sinkhole_list:
      if distance(point, sinkhole) < min_distance(sinkhole):
        valid = False
        break
    if valid: result.append(point)
  return result
  

def find_best_transform(
  ref_list,
  src_list,
  transform = None,
  iteration_scheme = None
  ):
  iteration_scheme = [(0.1, 0.0, 1000), (0.001, 0.000, 1000)] if iteration_scheme is None else iteration_scheme
  transform = Transform([0,-6],[[1,0], [0,1]], 0.1, 0.0) if transform is None else transform

  for db, dA, num_iteration in iteration_scheme:
    transform.db = db
    transform.dA = dA

    for i in xrange(num_iteration):
      moved_list = [transform.apply(v) for v in src_list]
      closest_pairs = compute_pairs(ref_list, moved_list)
      transform = icp_iteration(ref_list, moved_list, transform)
  return transform

