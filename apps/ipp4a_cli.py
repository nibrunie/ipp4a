#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Command Line Interface (CLI) for Image Pipeline Processing 4 Astronomy
# Copyrights Nicolas Brunie (2016-), nibrunie@gmail.com
# https://github.com/nibrunie/ipp4a
import argparse 

from lib.imageframe import ImageFrame
from lib.pointofinterest import *

parser = argparse.ArgumentParser(description='Image Processing Pipeline 4 Astronomy cli')
parser.add_argument('input_path', metavar='input', type=str, help='an input raw file to process', nargs='+')

args = parser.parse_args()

class AstronomyFrame:
  global_id = 0
  @staticmethod
  def get_new_id():
    new_id = AstronomyFrame.global_id
    AstronomyFrame.global_id += 1
    return new_id

  def __init__(self, input_path):
    self.frame_id = AstronomyFrame.get_new_id()

    print("processing {}.".format(input_path))
    self.input_img = ImageFrame.buildFromFile(input_path)
    downsized_img = self.input_img.resize(4, 4)
    #downsized_img.export("downsize.png")
    self.threshold_img = downsized_img.threshold(150)
    #threshold_img.export("threshold.png")
    self.poi_list = self.threshold_img.extract_poi(max_weight = 10**6, obj_per_part = 20)
    #superposed_img = threshold_img.poisuperposition(self.poi_list)
    #superposed_img.export("poi3.png")

frame_list = []

for input_path in args.input_path:
  new_frame = AstronomyFrame(input_path)
  frame_list.append(new_frame)

for new_frame in frame_list:
  new_frame.poi_list.sort(key = lambda v: v[2])
  print("poi_list, pre length is {}".format(len(new_frame.poi_list)))
  new_frame.reduced_list = exclude_sinkhole_neihbourg(new_frame.poi_list)
  print("reduced_list, length is {}".format(len(new_frame.reduced_list)))
  #new_frame.threshold_img.poisuperposition(reduced_list).export("poi{}.png".format(new_frame.frame_id))

print("finding best matching transform between two frames")
transform = Transform([0,0], [[1, 0], [0, 1]], 0.01, 0.01)
iteration_scheme = [(10.0, 0, 100), (1.0, 0.0, 100), (0.1, 0.0, 100), (0.001, 0.00, 100)] 

ref_list = frame_list[0].reduced_list
src_list = frame_list[1].reduced_list

len_list = min(len(ref_list), len(src_list))
ref_list = ref_list[:len_list]
src_list = src_list[:len_list]

closest_pairs = compute_pairs(ref_list, src_list)
closest_pairs.sort(key = lambda (u,v): -distance(u, v))
print closest_pairs[:10]


new_transform = find_best_transform(ref_list, src_list, transform, iteration_scheme)
print("new_transform: b={}, A={}".format(new_transform.b, new_transform.A))

moved_list = [new_transform.apply(v) for v in src_list]
closest_pairs = compute_pairs(ref_list, moved_list)
distance_list = [distance(p0, p1) for p0, p1 in closest_pairs]
print distance_list
average = sum(distance_list) / len(distance_list)
print("average distance is {}".format(average))

ref_frame   = frame_list[0].input_img
src_frame = frame_list[1].input_img
src_frame = src_frame.move(4 * new_transform.b[0], 4 * new_transform.b[1])
combined_img = ref_frame.combine(src_frame, 0.5, 0.5)
combined_img.export("combined.png")

#print("moving frame")
#main_frame.input_img.move(100.3, 37.17).export("moved.png")
