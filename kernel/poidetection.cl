/** Locate points of interest on input buffer @p img
 *
 */

/** 
 *           A tag is a 2D object stored in the object buffer,
 *           the object buffer has a two indirection levels. 
 *           A tag is an object which has itself as tag
 *
 * 1st step. Going throuh all the lines one by one, for each line
 *           determine the connex segments. 
 *           Each points of a connex segment points towards the upper
 *           leftmost points of this segment which became the segment
 *           object tag.
 *           This step ends up a one indirection tag. Each pixel is its
 *           own tag our points to its tag
 * 2nd step. Going through all the column, for each column determine
 *           the connex segments.
 *           For each connex segments, extract the segment tag of each pixel
 *           
 *
 *
 */           

float3 get_barycenter(float3 p0, float3 p1)
{
  float3 v;
  float w0 = p0.s2, w1 = p1.s2;
  v.x = (w0 * p0.x + w1 * p1.x) / (w0 + w1);
  v.y = (w0 * p0.y + w1 * p1.y) / (w0 + w1);
  v.s2 = w0 + w1;

  return v;
}

short2 get_pixel_object(__global short2*object, int x, int y, int h) {
  short2 tag = object[x * h + y];
  while ((tag.x != x || tag.y != y) && (tag.x >= 0) && (tag.y >= 0)) {
    x = tag.x; 
    y = tag.y;
    tag = object[x * h + y];
  }
  return tag;
}

void set_pixel_object(__global short2* object, int x, int y, int w, short2 tag)
{
  object[x * w + y] = tag;
}

__kernel void poidetection(
  __global const unsigned char* img, 
  __global short2* object, 
  __global float3* barycenter, 
  __global float3* results, 
  const int w, const int h, 
  const int sub_w, const int sub_h, 
  const int threshold, const int obj_per_partition)
{
  const int depth = 3;
  int work_id_x = get_global_id(0);
  int work_id_y = get_global_id(1);
  int start_x = work_id_x * sub_w;
  int start_y = work_id_y * sub_h;

  const int dim_x = depth * h;
  const int dim_y = depth;
  
  int x, y;

  {
    /** partition start index*/
    int pid = (work_id_x * 16 + work_id_y) * obj_per_partition;
    /** cancelling unused objects */
    for (int oid = 0; oid < obj_per_partition; ++oid) {
      results[pid + oid].s2 = -1.0f;
    }
    return;
  }

  /** start_obj is the line object initial coords */
  for (y = start_y; y < start_y + sub_h; ++y)
  {
    int start_obj = -1;
    for (x = start_x; x < start_x + sub_w; ++x)
    {
      unsigned char red   = img[x * dim_x + y * dim_y + 0]; 
      unsigned char green = img[x * dim_x + y * dim_y + 1]; 
      unsigned char blue  = img[x * dim_x + y * dim_y + 2]; 

      if (red >= threshold || green >= threshold || blue >= threshold) {
        if (start_obj < 0) {
          // starting a new object
          start_obj = x;
        } else {
          //last_obj = x;
        }
      } else {
        object[x * h + y] = (short2)(-1);
        // colors did not exceed threshold
        if (start_obj >= 0) {
          // this is the first pixel after a contiguous object
          // going from start_x to (x-1)
          int ox;
          short2 pixel_obj = (short2)(-1);
          pixel_obj.x = start_obj;
          pixel_obj.y = y;
          for (ox = start_obj; ox < x; ++ox) {
            object[ox * h + y] = pixel_obj;
          }
          barycenter[start_obj * h + y].x = (x + start_obj) / 2.0;
          barycenter[start_obj * h + y].y = (float) y;
          barycenter[start_obj * h + y].s2 = x - start_obj;
        }
        start_obj = -1;
      }
    }
    /** object ending line */
    if (start_obj >= 0) {
      int ox;
      short2 pixel_obj = (short2)(-1);
      pixel_obj.x = start_obj;
      pixel_obj.y = y;
      for (ox = start_obj; ox < x; ++ox) {
        object[ox * h + y] = pixel_obj;
      }
      barycenter[start_obj * h + y].x = (x + start_obj) / 2.0;
      barycenter[start_obj * h + y].y = (float) y;
      barycenter[start_obj * h + y].s2 = x - start_obj;
    }
  }

  /** processing column by column */
  for (x = start_x; x < start_x + sub_w; ++x)
  {
    int start_obj = -1;
    for (y = start_y; y < start_y + sub_h; ++y)
    {
      unsigned char red   = img[x * dim_x + y * dim_y + 0]; 
      unsigned char green = img[x * dim_x + y * dim_y + 1]; 
      unsigned char blue  = img[x * dim_x + y * dim_y + 2]; 

      if (red >= threshold || green >= threshold || blue >= threshold) {
        if (start_obj < 0) {
          // starting a new object
          start_obj = y;
        } else {
          //last_obj = x;
        }
      } else {
        // colors did not exceed threshold
        if (start_obj >= 0) 
        {
          // this is the first pixel after a contiguous object
          // going from start_x to (x-1)
          int oy;
          /** */
          short2 common_obj = get_pixel_object(object, x, start_obj, h);
          float3 new_barycenter = barycenter[x * h + start_obj];
          for (oy = start_obj; oy < y; ++oy) {
            short2 pixel_obj = get_pixel_object(object, x, oy, h);
            if (pixel_obj.x != common_obj.x || pixel_obj.y != common_obj.y) {
              new_barycenter = get_barycenter(new_barycenter, barycenter[x * h + oy]);
              set_pixel_object(object, pixel_obj.x, pixel_obj.y, h, common_obj);
            }
          }
          barycenter[common_obj.x * h + common_obj.y] = new_barycenter;
        }
        start_obj = -1;
      }
    }
    /** object ending line */
    if (start_obj >= 0) {
      // this is the first pixel after a contiguous object
      // going from start_obj to (y-1)
      int oy;
      /** */
      short2 common_obj = get_pixel_object(object, x, start_y, h);
      float3 new_barycenter = barycenter[x * h + start_obj];
      for (oy = start_obj; oy < y; ++oy) {
        short2 pixel_obj = get_pixel_object(object, x, oy, h);
        if (pixel_obj.x != common_obj.x || pixel_obj.y != common_obj.y) {
          new_barycenter = get_barycenter(new_barycenter, barycenter[x * h + oy]);
          set_pixel_object(object, pixel_obj.x, pixel_obj.y, h, common_obj);
        }
      }
      barycenter[common_obj.x * h + common_obj.y] = new_barycenter;
    }
  }

  /** partition start index*/
  int pid = (work_id_x * 16 + work_id_y) * obj_per_partition;
  /** cancelling unused objects */
  for (int oid = 0; oid < obj_per_partition; ++oid) {
    results[pid + oid].s2 = -1.0f;
  }
  /** object id */
  int oid = 0;
  for (int ox = start_x; ox < start_x + sub_w && oid < obj_per_partition; ++ ox) {
    for (int oy = start_y; oy < start_y + sub_h && oid < obj_per_partition; ++oy)
    {
      short2 pixel_obj = get_pixel_object(object, ox, oy, h);
      if (pixel_obj.x == ox && pixel_obj.y == y) {
        results[pid + oid] = barycenter[ox * h + oy];
        oid++;
      }
    }
  }

}
