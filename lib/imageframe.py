#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Image Frame module for Image Pipeline Processing 4 Astronomy
# Copyrights Nicolas Brunie (2016-), nibrunie@gmail.com
# https://github.com/nibrunie/ipp4a

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

import rawpy
import imageio
import sys

mf = cl.mem_flags

# Singleton/static class to contain
# OpenCL context and queue variable
class CLContext:
  ctx = None
  queue = None
  @staticmethod
  def get_context_queue():
    return CLContext.ctx, CLContext.queue
  @staticmethod
  def get_context():
    return CLContext.ctx
  @staticmethod
  def get_queue():
    return CLContext.queue

CLContext.ctx = cl.create_some_context()
CLContext.queue = cl.CommandQueue(CLContext.ctx)

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
    lin_size = reduce(lambda x,y: (x*y), self.get_shape(), 1)
    return self.md_array.reshape(lin_size).astype(self.dtype)

  def get_md_array(self):
    return self.md_array

  def get_cl_buffer_lin(self, flags = mf.READ_ONLY | mf.COPY_HOST_PTR):
    cl_ctx, cl_queue = CLContext.get_context_queue()
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
  def __init__(self, buffer, nbytes, dtype = np.uint8):
    cl_ctx, cl_queue = CLContext.get_context_queue()
    self.nbytes = nbytes
    self.buffer = buffer
    self.dtype = dtype

    self.cl_ctx = cl_ctx
    self.cl_queue = cl_queue

  def get_buffer(self):
    return self.buffer


class OffloadInputBuffer(OffloadBuffer):
  def __init__(self, raw_data, flags = mf.READ_ONLY):
    cl_ctx, cl_queue = CLContext.get_context_queue()
    dtype = raw_data.get_dtype()
    lin_data = raw_data.get_lin_array()
    nbytes = lin_data.shape[0] * np.dtype(dtype).itemsize
    buffer = cl.Buffer(cl_ctx, flags | mf.COPY_HOST_PTR, hostbuf = lin_data)
    OffloadBuffer.__init__(self, buffer, nbytes, dtype)

class OffloadInputOutputBuffer(OffloadInputBuffer):
  def __init__(self, raw_data, flags = mf.READ_WRITE):
    OffloadInputBuffer.__init__(self, raw_data, flags)

  def get_raw_data(self, shape):
    array = np.empty(self.nbytes / np.dtype(self.dtype).itemsize, dtype = self.dtype)
    cl.enqueue_copy(self.cl_queue, array, self.buffer)
    return RawData(array.reshape(shape).astype(self.dtype), dtype = self.dtype)


class OffloadOutputBuffer(OffloadBuffer):
  def __init__(self, nbytes, dtype = np.uint8, flags = mf.WRITE_ONLY):
    cl_ctx, cl_queue = CLContext.get_context_queue()
    buffer = cl.Buffer(cl_ctx, flags, nbytes)
    OffloadBuffer.__init__(self, buffer, nbytes, dtype)

  def get_raw_data(self, shape):
    array = np.empty(self.nbytes / np.dtype(self.dtype).itemsize, dtype = self.dtype)
    cl.enqueue_copy(self.cl_queue, array, self.buffer)
    return RawData(array.reshape(shape).astype(self.dtype), dtype = self.dtype)


class OffloadProcess(object):
  kernel_map = {}
  default_kernels = ["kernel/combine.cl", "kernel/dark.cl", "kernel/downsize.cl", "kernel/poidetection.cl", "kernel/threshold.cl"]
  @staticmethod
  def getOffloadProcess(filename):
    if not filename in OffloadProcess.kernel_map:
      OffloadProcess.kernel_map[filename] = OffloadProcess.create_from_kernel_filename(filename)
    return OffloadProcess.kernel_map[filename]

  @staticmethod
  def preLoadKernels():
    klist = OffloadProcess.default_kernels
    for kernel_path in klist:
      OffloadProcess.getOffloadProcess(kernel_path)

  def __init__(self, cl_prg):
    cl_ctx, cl_queue = CLContext.get_context_queue()
    self.cl_ctx = cl_ctx
    self.cl_queue = cl_queue
    self.cl_prg = cl_prg

  def get_cl_prg(self):
    return self.cl_prg

  @staticmethod
  def create_from_kernel_src(kernel_src):
    cl_ctx = CLContext.get_context()
    kernel_prg = cl.Program(cl_ctx, kernel_src).build()
    return OffloadProcess(kernel_prg)

  @staticmethod
  def create_from_kernel_filename(kernel_filename):
    print("creating kernel from {}".format(kernel_filename))
    kernel_src = open(kernel_filename, "r").read()
    return OffloadProcess.create_from_kernel_src(kernel_src)

class ImageFrame:
  def __init__(self, raw_data):
    self.raw_data = raw_data

  def get_shape(self):
    return self.raw_data.get_shape()

  def get_raw_data(self):
    return self.raw_data

  @staticmethod
  def buildFromFile(filename):
    return ImageFrame(RawData.build_from_raw_file(filename))

  def resize(self, div_w, div_h):
    w, h, d = self.get_shape()
    out_w = w / div_w
    out_h = h / div_h
    out_buffer_size = out_w * out_h * d * np.dtype(self.raw_data.dtype).itemsize 

    img_buffer = OffloadInputBuffer(self.raw_data)
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_downsize = OffloadProcess.getOffloadProcess("kernel/downsize.cl")

    kernel_downsize.get_cl_prg().downsize(
      CLContext.get_queue(),
      (out_w, out_h),
      None,
      img_buffer.get_buffer(),
      out_buffer.get_buffer(),
      np.int32(w),
      np.int32(h),
      np.int32(div_w),
      np.int32(div_h)
    )
    return ImageFrame(out_buffer.get_raw_data((out_w, out_h, d)))

  def threshold(self, threshold = 100):
    w, h, d = self.get_shape()
    out_buffer_size = w * h * d * np.dtype(self.raw_data.dtype).itemsize 

    img_buffer = OffloadInputBuffer(self.raw_data)
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_threshold = OffloadProcess.getOffloadProcess("kernel/threshold.cl")

    kernel_threshold.get_cl_prg().threshold(
      CLContext.get_queue(), 
      (w, h), 
      None, 
      img_buffer.get_buffer(), 
      out_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(1), 
      np.int32(1), 
      np.int32(threshold)
    )
    return ImageFrame(out_buffer.get_raw_data((w, h, d)))

  ## Extract Points of interest
  def extract_poi(self, 
        obj_per_part = 5, 
        poi_part_nx = 16, 
        poi_part_ny = 16, 
        threshold = 100,
        min_weight = 5.0,
        max_weight = 50.0):
    w,h,d = self.get_shape()
    poi_part_w = w / poi_part_nx
    poi_part_h = h / poi_part_ny
    poi_size = obj_per_part * poi_part_nx * poi_part_ny * 4 * np.dtype(np.float32).itemsize

    img_buffer        = OffloadInputOutputBuffer(self.raw_data)
    object_buffer     = OffloadOutputBuffer(w * h * np.dtype(np.int16).itemsize * 2, flags = mf.READ_WRITE, dtype = np.int16)
    barycenter_buffer = OffloadOutputBuffer(w * h * np.dtype(np.float32).itemsize * 4, dtype = np.float32, flags = mf.READ_WRITE)
    poi_buffer        = OffloadOutputBuffer(poi_size, dtype = np.float32)
    kernel_poi_detection = OffloadProcess.getOffloadProcess("kernel/poidetection.cl")

    completeEvent = kernel_poi_detection.get_cl_prg().poidetection(
      CLContext.get_queue(), 
      (poi_part_nx, poi_part_ny), 
      None, 
      img_buffer.get_buffer(), 
      object_buffer.get_buffer(), 
      barycenter_buffer.get_buffer(), 
      poi_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(poi_part_w), 
      np.int32(poi_part_h), 
      np.int32(threshold), 
      np.int32(obj_per_part)
    )
    completeEvent.wait()

    completeEvent = kernel_poi_detection.get_cl_prg().poiunification(
      CLContext.get_queue(), 
      (1, 1), 
      None, 
      img_buffer.get_buffer(), 
      object_buffer.get_buffer(), 
      barycenter_buffer.get_buffer(), 
      poi_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(poi_part_w), 
      np.int32(poi_part_h), 
      np.int32(threshold), 
      np.int32(obj_per_part)
    )
    completeEvent.wait()

    completeEvent = kernel_poi_detection.get_cl_prg().poiextraction(
      CLContext.get_queue(), 
      (poi_part_nx, poi_part_ny), 
      None, 
      img_buffer.get_buffer(), 
      object_buffer.get_buffer(), 
      barycenter_buffer.get_buffer(), 
      poi_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(poi_part_w), 
      np.int32(poi_part_h), 
      np.int32(threshold), 
      np.int32(obj_per_part)
    )
    completeEvent.wait()

    point_list = []

    poi_data = poi_buffer.get_raw_data((poi_part_nx * poi_part_ny * obj_per_part * 4,))
    for i in xrange(poi_part_nx * poi_part_ny * obj_per_part):
      obj_x, obj_y, obj_w = poi_data[i * 4], poi_data[i * 4 + 1], poi_data[i * 4 + 2]
      if obj_w >= min_weight and obj_w <= max_weight:
        point_list.append((obj_x, obj_y, obj_w, 0.0))
    return point_list

  def poisuperposition(self, poi_list):
    w,h,d = self.get_shape()

    num_obj = len(poi_list)
    poi_array = np.array(poi_list).astype(np.float32).reshape((num_obj * 4,))

    poi_data = RawData(poi_array, dtype = np.float32)

    img_buffer        = OffloadInputOutputBuffer(self.raw_data)
    poi_buffer        = OffloadInputBuffer(poi_data)
    kernel_poi_detection = OffloadProcess.getOffloadProcess("kernel/poidetection.cl")

    completeEvent = kernel_poi_detection.get_cl_prg().poisuperposition(
      CLContext.get_queue(), 
      (1, 1), 
      None, 
      img_buffer.get_buffer(), 
      poi_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(num_obj)
    )
    completeEvent.wait()
    return ImageFrame(img_buffer.get_raw_data((w, h, d)))

  ## Combine @p self ImageFormat with @p source
  # ImageFormat object using
  # dest = c0 x self + c1 x source
  def combine(self, source, c0, c1):
    w, h, d = self.get_shape()
    out_buffer_size = w * h * d * np.dtype(self.raw_data.dtype).itemsize 

    img0_buffer = OffloadInputBuffer(self.raw_data)
    img1_buffer = OffloadInputBuffer(source.raw_data)
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_combine = OffloadProcess.getOffloadProcess("kernel/combine.cl")

    kernel_combine.get_cl_prg().combine(
      CLContext.get_queue(), 
      (w, h), 
      None, 
      img0_buffer.get_buffer(), 
      img1_buffer.get_buffer(), 
      out_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(1), 
      np.int32(1), 
      np.float32(c0),
      np.float32(c1)
    )
    return ImageFrame(out_buffer.get_raw_data((w, h, d)))

  ## move @p self ImageFormat according to
  #  @p dx, @p dy move vector
  def move(self, dx, dy):
    w, h, d = self.get_shape()
    out_buffer_size = w * h * d * np.dtype(self.raw_data.dtype).itemsize 

    img_buffer = OffloadInputBuffer(self.raw_data)
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_combine = OffloadProcess.getOffloadProcess("kernel/combine.cl")

    kernel_combine.get_cl_prg().move(
      CLContext.get_queue(), 
      (w, h), 
      None, 
      img_buffer.get_buffer(), 
      out_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(1), 
      np.int32(1), 
      np.float32(dx),
      np.float32(dy)
    )
    return ImageFrame(out_buffer.get_raw_data((w, h, d)))

  def rotatePiOver2(self, m = 1):
    return ImageFrame(RawData(np.rot90(self.get_raw_data().get_md_array(), m)))

  ## Compute the maximum between @p self image and 
  #  @p dark_img
  #  @return an ImageFrame containing the dark interpolation of both inputs
  def update_dark(self, dark_img):
    new_array = np.maximum(self.get_raw_data().get_md_array(), dark_img.get_raw_data().get_md_array())
    return ImageFrame(RawData(new_array))

  ## Delete the dark portion of @p self which is 
  #  described by @p dark_img
  #  @return clean version of self
  def clean_dark(self, dark_img, dark_threshold = 50.0):
    w, h, d = self.get_shape()
    out_buffer_size = w * h * d * np.dtype(self.raw_data.dtype).itemsize 

    img_buffer = OffloadInputBuffer(self.get_raw_data())
    dark_buffer = OffloadInputBuffer(dark_img.get_raw_data())
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_dark = OffloadProcess.getOffloadProcess("kernel/dark.cl")

    kernel_dark.get_cl_prg().clean_dark(
      CLContext.get_queue(), 
      (w, h), 
      None, 
      img_buffer.get_buffer(), 
      dark_buffer.get_buffer(),
      out_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(1), 
      np.int32(1),
      np.float32(dark_threshold)
    )
    return ImageFrame(out_buffer.get_raw_data((w, h, d)))

  def subtract_threshold(self, dark_threshold):
    w, h, d = self.get_shape()
    out_buffer_size = w * h * d * np.dtype(self.raw_data.dtype).itemsize 

    img_buffer = OffloadInputBuffer(self.get_raw_data())
    out_buffer = OffloadOutputBuffer(out_buffer_size)
    kernel_dark = OffloadProcess.getOffloadProcess("kernel/dark.cl")

    kernel_dark.get_cl_prg().subtract_threshold(
      CLContext.get_queue(), 
      (w, h), 
      None, 
      img_buffer.get_buffer(), 
      np.float32(dark_threshold),
      out_buffer.get_buffer(), 
      np.int32(w), 
      np.int32(h), 
      np.int32(1), 
      np.int32(1)
    )
    return ImageFrame(out_buffer.get_raw_data((w, h, d)))

  def export(self, filename):
    imageio.imsave(filename, self.raw_data.get_md_array())
    







