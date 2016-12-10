/** Downsize the input image @p img of size @p w x @p h
 *  to the output image @p out of size global_size.x x global_size.y
 *  Each work item processes a sub-buffer of size @p sub_w x @p sub_h
 */
__kernel void threshold(__global const unsigned char* img, __global unsigned char* out, const int w, const int h, const int sub_w, const int sub_h, const int threshold)
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
      unsigned char red = img[x * dim_x + y * dim_y + 0];
      unsigned char green = img[x * dim_x + y * dim_y + 1];
      unsigned char blue = img[x * dim_x + y * dim_y + 2];
      if (red >= threshold || green >= threshold || blue >= threshold) 
      {
        out[x * dim_x + y * dim_y + 0] = 255;
        out[x * dim_x + y * dim_y + 1] = 255;
        out[x * dim_x + y * dim_y + 2] = 255;
      } else {
        out[x * dim_x + y * dim_y + 0] = 0;
        out[x * dim_x + y * dim_y + 1] = 0;
        out[x * dim_x + y * dim_y + 2] = 0;
      }
  }

}
