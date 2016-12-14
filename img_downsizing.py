#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

import rawpy
import imageio
import sys

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

class RawData(object):
  def __init__(self, md_array, dtype = np.uint8):
    self.md_array = md_array
    self.dtype = dtype

  def __getitem__(self, id):
    return self.md_array[id]

  def get_shape(self):
    return self.md_array.shape
  def get_dtype(self):
    return self.dtype

  def get_lin_array(self):
    w, h, d = self.get_shape()
    return self.md_array.reshape(w*h*d).astype(self.dtype)

  def get_md_array(self):
    return self.md_array

  def get_cl_buffer_lin(self, cl_ctx, cl_queue, flags = mf.READ_ONLY | mf.COPY_HOST_PTR):
    return cl.Buffer(ctx, flags, host_buf = self.get_lin_array())

  @staticmethod
  ## Read a Raw file and extract 
  #  image data under the shape of a numpy (w, h, 3)
  #  shaped array
  def build_from_raw_file(input_path):
    raw = rawpy.imread(input_path)
    rgb = raw.postprocess()
    return RawData(rgb)
  
class OffloadBuffer(object):
  def __init__(self, cl_ctx, cl_queue, buffer, nbytes, dtype = np.uint8):
    self.nbytes = nbytes
    self.buffer = buffer
    self.dtype = dtype

    self.cl_ctx = cl_ctx
    self.cl_queue = cl_queue

  def get_buffer(self):
    return self.buffer


class OffloadInputBuffer(OffloadBuffer):
  def __init__(self, cl_ctx, cl_queue, raw_data):
    dtype = raw_data.get_dtype()
    lin_data = raw_data.get_lin_array()
    nbytes = lin_data.shape[0] * np.dtype(dtype).itemsize
    buffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = lin_data)
    OffloadBuffer.__init__(self, cl_ctx, cl_queue, buffer, nbytes, dtype)

class OffloadOutputBuffer(OffloadBuffer):
  def __init__(self, cl_ctx, cl_queue, nbytes, dtype = np.uint8, flags = mf.WRITE_ONLY):
    buffer = cl.Buffer(cl_ctx, flags, nbytes)
    OffloadBuffer.__init__(self, cl_ctx, cl_queue, buffer, nbytes, dtype)

  def get_raw_data(self, shape):
    array = np.empty(self.nbytes / np.dtype(self.dtype).itemsize, dtype = self.dtype)
    cl.enqueue_copy(self.cl_queue, array, self.buffer)
    return RawData(array.reshape(shape).astype(self.dtype), dtype = self.dtype)


class OffloadProcess(object):
  def __init__(self, cl_ctx, cl_queue, cl_prg):
    self.cl_ctx = cl_ctx
    self.cl_queue = cl_queue
    self.cl_prg = cl_prg

  def get_cl_prg(self):
    return self.cl_prg

  @staticmethod
  def create_from_kernel_src(cl_ctx, cl_queue, kernel_src):
    kernel_prg = cl.Program(cl_ctx, kernel_src).build()
    return OffloadProcess(cl_ctx, cl_queue, kernel_prg)

  @staticmethod
  def create_from_kernel_filename(cl_ctx, cl_queue, kernel_filename):
    kernel_src = open(kernel_filename, "r").read()
    return OffloadProcess.create_from_kernel_src(cl_ctx, cl_queue, kernel_src)



input_path = sys.argv[1]

image_raw = RawData.build_from_raw_file(input_path)
img_downsize = OffloadProcess.create_from_kernel_filename(ctx, queue, "kernel/downsize.cl")
img_threshold = OffloadProcess.create_from_kernel_filename(ctx, queue, "kernel/threshold.cl")
img_poidetection = OffloadProcess.create_from_kernel_filename(ctx, queue, "kernel/poidetection.cl")

w,h,d = image_raw.get_shape()
print("w={}, h={}, d={}\n".format(w, h, d))
div_w = 4
div_h = 4
out_w = w / div_w
out_h = h / div_h
type_size = np.dtype(np.uint8).itemsize
# rgb_lin = image_raw.get_lin_array()


# rgb_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = rgb_lin)
image_buffer = OffloadInputBuffer(ctx, queue, image_raw)

tmp_buffer = OffloadOutputBuffer(ctx, queue, out_w * out_h * d * type_size, flags = mf.READ_WRITE)

out_buffer = OffloadOutputBuffer(ctx, queue, out_w * out_h * d * type_size, flags = mf.READ_WRITE)



# res_g = cl.Buffer(ctx, mf.WRITE_ONLY, out_w * out_h * d * type_size)
print("downsizing image")
img_downsize.get_cl_prg().mysample(queue, (out_w, out_h), None, image_buffer.get_buffer(), tmp_buffer.get_buffer(), np.int32(w), np.int32(h), np.int32(div_w), np.int32(div_h))
print("threshold simplification")
img_threshold.get_cl_prg().threshold(queue, (out_w, out_h), None, tmp_buffer.get_buffer(), out_buffer.get_buffer(), np.int32(out_w), np.int32(out_h), np.int32(1), np.int32(1), np.int32(100))

print("post-processing")
output_rawdata = out_buffer.get_raw_data((out_w, out_h, 3))
print("final rendering")
imageio.imsave("sample.jpg", output_rawdata.get_md_array())


poi_part_nx = 16
poi_part_ny = 16
poi_part_w = out_w / poi_part_nx
poi_part_h = out_h / poi_part_ny
obj_per_part = 5
poi_size = obj_per_part * poi_part_nx * poi_part_ny * 3 * np.dtype(np.float32).itemsize
print("poi_size= {}".format(poi_size))

object_buffer = OffloadOutputBuffer(ctx, queue, out_w * out_h * np.dtype(np.int16).itemsize * 2, flags = mf.READ_WRITE, dtype = np.int16)
barycenter_buffer = OffloadOutputBuffer(ctx, queue, out_w * out_h * np.dtype(np.float32).itemsize * 3, dtype = np.float32, flags = mf.READ_WRITE)
poi_buffer = OffloadOutputBuffer(ctx, queue, poi_size, dtype = np.float32)

print("point of interest detection")
print("{} x {} -> {} x {}".format(out_w, out_h, poi_part_w, poi_part_h))
img_poidetection.get_cl_prg().poidetection(
  queue, 
  (poi_part_nx, poi_part_ny), 
  None, 
  out_buffer.get_buffer(), 
  object_buffer.get_buffer(), 
  barycenter_buffer.get_buffer(), 
  poi_buffer.get_buffer(), 
  np.int32(out_w), 
  np.int32(out_h), 
  np.int32(poi_part_w), 
  np.int32(poi_part_h), 
  np.int32(200), 
  np.int32(obj_per_part)
)



if 0:
  # there are 3 float for each results data
  poi_data = poi_buffer.get_raw_data((poi_part_nx * poi_part_ny * obj_per_part * 3,))
  print("poi_data.shape {}".format(poi_data.md_array.shape))
  for i in xrange(poi_part_nx * poi_part_ny * obj_per_part):
    obj_x, obj_y, obj_w = poi_data[i * 3], poi_data[i * 3 + 1], poi_data[i * 3 + 2]
    if obj_w >= 1.0:
      print("object at {},{} with weight {}".format(obj_x, obj_y, obj_w))


