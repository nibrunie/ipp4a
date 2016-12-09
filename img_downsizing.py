#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

import rawpy
import imageio
import sys

input_path = sys.argv[1]
raw = rawpy.imread(input_path)
rgb = raw.postprocess()
w,h,d = rgb.shape
print(rgb.dtype)
rgb_lin = rgb.reshape(w*h*d).astype(np.uint8)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

rgb_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = rgb_lin)

prg = cl.Program(ctx, """
__kernel void mysample(__global const unsigned char* img, __global unsigned char* out, const int w, const int h, const int sub_w, const int sub_h)
{
  const int depth = 3;
  const int dim_x = depth * h, dim_y = depth;
  const int out_dim_x = depth * get_global_size(1), out_dim_y = depth;

  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int start_x = gid_x * sub_w;
  int start_y = gid_y * sub_h;
  int x,y;
  float acc = 0.0f;
  float factor = 1 / (3.0f * sub_w * sub_h);
  for (x = start_x; x < start_x + sub_w; ++x)
    for (y = start_y; y < start_y + sub_h; ++y) {
      acc += (convert_float(img[x * dim_x + y * dim_y + 0]) + 
              convert_float(img[x * dim_x + y * dim_y + 1]) +  
              convert_float(img[x * dim_x + y * dim_y + 2])) * factor;  
  }
  out[gid_x * out_dim_x + gid_y * out_dim_y + 0] = convert_uchar_sat(acc);
  out[gid_x * out_dim_x + gid_y * out_dim_y + 1] = convert_uchar_sat(acc);
  out[gid_x * out_dim_x + gid_y * out_dim_y + 2] = convert_uchar_sat(acc);

}
""").build()

div_w = 4
div_h = 4

out_w = w / div_w
out_h = h / div_h
type_size = np.dtype(np.uint8).itemsize
print("w,h,out_w,out_h: ({},{},{},{})".format(w, h, out_w, out_h))
print(w)
print(h)

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, out_w * out_h * d * type_size)
prg.mysample(queue, (out_w, out_h), None, rgb_b, res_g, np.int32(w), np.int32(h), np.int32(div_w), np.int32(div_h))

res_np = np.empty(out_w*out_h*d, dtype=np.uint8)
cl.enqueue_copy(queue, res_np, res_g)
res_np = res_np.reshape((out_w, out_h, 3))
print(res_np)
print(res_np.shape)
imageio.imsave("sample.jpg", res_np)

