/** Downsize the input image @p img of size @p w x @p h
 *  to the output image @p out of size global_size.x x global_size.y
 *  Each work item processes a sub-buffer of size @p sub_w x @p sub_h
 */
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
