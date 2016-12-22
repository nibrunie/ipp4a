
/** Downsize the input image @p img of size @p w x @p h
 *  to the output image @p out of size global_size.x x global_size.y
 *  Each work item processes a sub-buffer of size @p sub_w x @p sub_h
 */
__kernel void combine(
  __global const unsigned char* img0, 
  __global const unsigned char* img1, 
  __global unsigned char* out, 
  const int w, 
  const int h, 
  const int sub_w, 
  const int sub_h, 
  const float c0,
  const float c1
  )
{
  const int depth = 3;
  const int dim_x = depth * h, dim_y = depth;

  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int start_x = gid_x * sub_w;
  int start_y = gid_y * sub_h;
  int x,y;

  for (x = start_x; x < start_x + sub_w; ++x)
    for (y = start_y; y < start_y + sub_h; ++y) { 
      unsigned char red0    = img0[x * dim_x + y * dim_y + 0];
      unsigned char green0  = img0[x * dim_x + y * dim_y + 1];
      unsigned char blue0   = img0[x * dim_x + y * dim_y + 2];
      unsigned char red1    = img1[x * dim_x + y * dim_y + 0];
      unsigned char green1  = img1[x * dim_x + y * dim_y + 1];
      unsigned char blue1   = img1[x * dim_x + y * dim_y + 2];
      out[x * dim_x + y * dim_y + 0] = c0 * red0 + c1 * red1;
      out[x * dim_x + y * dim_y + 1] = c0 * green0 + c1 * green1;
      out[x * dim_x + y * dim_y + 2] = c0 * blue0 + c1 * blue1;
  }
}

uchar3 get_color(__global const unsigned char* img, int w, int h, int depth, int x, int y)
{
  const int dim_x = depth * h, dim_y = depth;
  uchar3 result;
  if (x < 0 || x >= w || y < 0 || y >= h ) return (uchar3)(0);
  result.x = img[x * dim_x + y * dim_y + 0];
  result.y = img[x * dim_x + y * dim_y + 1];
  result.z = img[x * dim_x + y * dim_y + 2];
  return result;
}

__kernel void move(
  __global const unsigned char* img, 
  __global unsigned char* out, 
  const int w, 
  const int h, 
  const int sub_w, 
  const int sub_h, 
  const float dx,
  const float dy
  )
{
  const int depth = 3;
  const int dim_x = depth * h, dim_y = depth;

  const int ix = ceil(dx);
  const int iy = ceil(dy);
  const float fx = dx - ix;
  const float fy = dy - iy;

  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int start_x = gid_x * sub_w;
  int start_y = gid_y * sub_h;
  int x,y;

  for (x = start_x; x < start_x + sub_w; ++x)
    for (y = start_y; y < start_y + sub_h; ++y) { 
      uchar3 red00 = get_color(img, w, h, depth, x - ix, y - iy); 
      uchar3 red01 = get_color(img, w, h, depth, x - ix + 1, y - iy); 
      uchar3 red10 = get_color(img, w, h, depth, x - ix, y - iy + 1); 
      uchar3 red11 = get_color(img, w, h, depth, x - ix + 1, y - iy + 1); 

      float3 color = (fx * fy) * convert_float3(red00) 
          + (1-fx) * fy * convert_float3(red01)
          + fx * (1 - fy) * convert_float3(red10)
          + (1 - fx) * (1 - fy) * convert_float3(red11);

      uchar3 ucolor = convert_uchar3_sat(color);
      out[x * dim_x + y * dim_y + 0] = ucolor.x;
      out[x * dim_x + y * dim_y + 1] = ucolor.y;
      out[x * dim_x + y * dim_y + 2] = ucolor.z;
  }
}
