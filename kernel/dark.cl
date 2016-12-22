

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

uchar3 clean_color(uchar3 pixel_color, uchar3 dark_color)
{
  float3 pixel_color_f = convert_float3(pixel_color);
  float3 dark_color_f  = convert_float3(dark_color);

  pixel_color_f -= dark_color_f;
  float3 scaling_factor = (float3) (255.0f) / ((float3) (255.0f) - dark_color_f);
  pixel_color_f *= scaling_factor;
  return convert_uchar3_sat(pixel_color_f);
}

__kernel void clean_dark(
  __global const unsigned char* img, 
  __global const unsigned char* dark_img, 
  __global unsigned char* out, 
  const int w, 
  const int h, 
  const int sub_w, 
  const int sub_h,
  const float dark_threshold
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
      uchar3 pixel_color = get_color(img, w, h, depth, x, y);
      uchar3 dark_color = get_color(dark_img, w, h, depth, x, y);
      uchar3 final_color = clean_color(pixel_color, dark_color);

      out[x * dim_x + y * dim_y + 0] = final_color.x;
      out[x * dim_x + y * dim_y + 1] = final_color.y;
      out[x * dim_x + y * dim_y + 2] = final_color.z;
  }
}

__kernel void subtract_threshold(
  __global const unsigned char* img, 
  const float threshold_red,
  const float threshold_green,
  const float threshold_blue,
  __global unsigned char* out, 
  const int w, 
  const int h, 
  const int sub_w, 
  const int sub_h
  )
{
  const int depth = 3;
  const int dim_x = depth * h, dim_y = depth;

  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);

  int start_x = gid_x * sub_w;
  int start_y = gid_y * sub_h;
  int x,y;
  uchar3 dark_color = convert_uchar3_sat((float3) (threshold_red, threshold_green, threshold_blue));

  for (x = start_x; x < start_x + sub_w; ++x)
    for (y = start_y; y < start_y + sub_h; ++y) { 
      uchar3 pixel_color = get_color(img, w, h, depth, x, y);
      uchar3 final_color = clean_color(pixel_color, dark_color);

      out[x * dim_x + y * dim_y + 0] = final_color.x;
      out[x * dim_x + y * dim_y + 1] = final_color.y;
      out[x * dim_x + y * dim_y + 2] = final_color.z;
  }
}
