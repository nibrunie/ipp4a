#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Static Pipeline for Image Pipeline Processing 4 Astronomy
# Copyrights Nicolas Brunie (2016-), nibrunie@gmail.com
# https://github.com/nibrunie/ipp4a

from lib.imageframe import ImageFrame, OffloadProcess
from lib.pointofinterest import *
import numpy as np

dark_paths = [
  "./Astronomie/PC020227.ORF",
#  "./Astronomie/PC020228.ORF",
#  "./Astronomie/PC020229.ORF",
#  "./Astronomie/PC020230.ORF",
#  "./Astronomie/PC020231.ORF",
#  "./Astronomie/PC020232.ORF",
#  "./Astronomie/PC020233.ORF",
#  "./Astronomie/PC020234.ORF",
#  "./Astronomie/PC020235.ORF",
#  "./Astronomie/PC020236.ORF",
]

OffloadProcess.preLoadKernels()

class AstronomyFrame:
  global_id = 0
  @staticmethod
  def get_new_id():
    new_id = AstronomyFrame.global_id
    AstronomyFrame.global_id += 1
    return new_id

  def __init__(self, input_path):
    self.frame_id = AstronomyFrame.get_new_id()

    print("importing {}.".format(input_path))
    self.image_frame = ImageFrame.buildFromFile(input_path)

dark_list = []

for input_path in dark_paths:
  new_frame = AstronomyFrame(input_path)
  dark_list.append(new_frame)

initial_dark = dark_list[0].image_frame
initial_dark.export("initial_dark.png")
for dark_frame in dark_list[1:]:
  initial_dark.update_dark(dark_frame.image_frame)

initial_dark.rotatePiOver2().export("dark.png")
# dark_average = np.average(initial_dark.get_raw_data().get_md_array())
dark_average = initial_dark.average_per_channel()
print("dark average is {}".format(dark_average))

light_frame = AstronomyFrame("./Astronomie/PC010192.ORF")
clean_frame = light_frame.image_frame.subtract_threshold(dark_average[0], dark_average[1], dark_average[2])
clean_frame.export("clean_frame.png")
