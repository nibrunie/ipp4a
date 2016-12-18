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

float4 get_barycenter(float4 p0, float4 p1)
{
  float4 v;
  float w0 = fabs(p0.z), w1 = fabs(p1.z);
  v.x = (w0 * p0.x + w1 * p1.x) / (w0 + w1);
  v.y = (w0 * p0.y + w1 * p1.y) / (w0 + w1);
  v.z = p0.z + p1.z;

  return v;
}

short2 get_pixel_object(__global short2*object, int x, int y, int h) {
  short2 tag = object[x * h + y];
  int ox = x, oy = y;
  while ((tag.x != ox || tag.y != oy) && (tag.x >= 0) && (tag.y >= 0)) {
    ox = tag.x; 
    oy = tag.y;
    tag = object[ox * h + oy];
  }
  return tag;
}

void set_pixel_object(__global short2* object, int x, int y, int h, short2 tag)
{
  object[x * h + y] = tag;
}

__kernel void poidetection(
  __global unsigned char* img, 
  __global short2* object, 
  __global float4* barycenter, 
  __global float4* results, 
  const int w, const int h, 
  const int sub_w, const int sub_h, 
  const int threshold, const int obj_per_partition)
{
  const int depth = 3;
  int work_id_x = get_global_id(0);
  int work_id_y = get_global_id(1);
  const int dim_x = depth * h;
  const int dim_y = depth;

  int start_x = work_id_x * sub_w;
  int start_y = work_id_y * sub_h;
  
  int x, y;

  /** start_obj is the line object initial coords */
  for (y = start_y; y < start_y + sub_h; ++y)
  {
    short2 start_tag = (short2)(-1, -1);
    float weight = 0.0f;
    for (x = start_x; x < start_x + sub_w; ++x)
    {
      unsigned char red   = img[x * dim_x + y * dim_y + 0]; 
      unsigned char green = img[x * dim_x + y * dim_y + 1]; 
      unsigned char blue  = img[x * dim_x + y * dim_y + 2]; 


      if (red >= threshold || green >= threshold || blue >= threshold) {
        if (start_tag.x < 0) {
          // starting a new object
          start_tag = (short2) (x, y);
          barycenter[start_tag.x * h + y].y = y;
          weight = 0.0f;
        } 
        object[x * h + y] = start_tag;
        barycenter[start_tag.x * h + y].x = (x + start_tag.x) * 0.5;
        weight += 1.0f;
        // offset for cells touching the boundaries
        //if (x == start_x + sub_w - 1 || y == start_y + sub_h - 1 || x == start_x || y == start_y)
         // weight -= w * h;
        barycenter[start_tag.x * h + y].z = weight; // x - start_tag.x + 1.0;
      } else {
        start_tag = (short2) (-1, -1);
        object[x * h + y] = start_tag;
      }
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
      short2 common_obj;
      float4 new_barycenter;
      if (red >= threshold || green >= threshold || blue >= threshold) {
        if (start_obj < 0) {
          // starting a new object
          start_obj = y;
          common_obj = get_pixel_object(object, x, y, h);
          new_barycenter = barycenter[common_obj.x * h + common_obj.y];
        } else {
          short2 pixel_obj = get_pixel_object(object, x, y, h);
          if (pixel_obj.x != common_obj.x || pixel_obj.y != common_obj.y) {
            new_barycenter = get_barycenter(new_barycenter, barycenter[pixel_obj.x * h + pixel_obj.y]);
            set_pixel_object(object, pixel_obj.x, pixel_obj.y, h, common_obj);
            barycenter[common_obj.x * h + common_obj.y] = new_barycenter;
          }
        }
      } else {
        // colors did not exceed threshold
        if (start_obj >= 0) start_obj = -1; 
      }
    }
  }

}

__kernel void poiunification(
  __global unsigned char* img, 
  __global short2* object, 
  __global float4* barycenter, 
  __global float4* results, 
  const int w, const int h, 
  const int sub_w, const int sub_h, 
  const int threshold, const int obj_per_partition)
{
  const int depth = 3;
  int work_id_x = get_global_id(0);
  int work_id_y = get_global_id(1);

  const int dim_x = depth * h;
  const int dim_y = depth;
  

  /** processing each column separating partitions
   *  from poidetection kernel execution */
  for (int x = sub_w-1; x < w; x += sub_w)
  {
    for (int y = 0; y < h; ++y)
    {
      // pixel (x, y)
      unsigned char red0   = img[x * dim_x + y * dim_y + 0]; 
      unsigned char green0 = img[x * dim_x + y * dim_y + 1]; 
      unsigned char blue0  = img[x * dim_x + y * dim_y + 2]; 
      // pixel (x+1, y)
      unsigned char red1   = img[(x + 1) * dim_x + y * dim_y + 0]; 
      unsigned char green1 = img[(x + 1) * dim_x + y * dim_y + 1]; 
      unsigned char blue1  = img[(x + 1) * dim_x + y * dim_y + 2]; 

      if ((red0 >= threshold || green0 >= threshold || blue0 >= threshold) &&
          (red1 >= threshold || green1 >= threshold || blue1 >= threshold))
      {
        short2 obj0 = get_pixel_object(object, x, y, h);
        short2 obj1 = get_pixel_object(object, x + 1, y, h);
        if (obj0.x != obj1.x || obj0.y != obj1.y) {
          float4 new_barycenter = get_barycenter(
            barycenter[obj0.x * h + obj0.y],
            barycenter[obj1.x * h + obj1.y]
          );
          set_pixel_object(object, obj1.x, obj1.y, h, obj0);
          barycenter[obj0.x * h + obj0.y] = new_barycenter;
        }
      }
    }
  }


  /** processing each lines separating partitions
   *  from poidetection kernel execution */
  for (int y = sub_h-1; y < h; y += sub_h)
  {
    for (int x = 0; x < w; ++x)
    {
      // pixel (x, y)
      unsigned char red0   = img[x * dim_x + y * dim_y + 0]; 
      unsigned char green0 = img[x * dim_x + y * dim_y + 1]; 
      unsigned char blue0  = img[x * dim_x + y * dim_y + 2]; 
      // pixel (x+1, y)
      unsigned char red1   = img[x * dim_x + (y + 1) * dim_y + 0]; 
      unsigned char green1 = img[x * dim_x + (y + 1) * dim_y + 1]; 
      unsigned char blue1  = img[x * dim_x + (y + 1) * dim_y + 2]; 

      if ((red0 >= threshold || green0 >= threshold || blue0 >= threshold) &&
          (red1 >= threshold || green1 >= threshold || blue1 >= threshold))
      {
        short2 obj0 = get_pixel_object(object, x, y, h);
        short2 obj1 = get_pixel_object(object, x, y+1, h);
        if (obj0.x != obj1.x || obj0.y != obj1.y) {
          float4 new_barycenter = get_barycenter(
            barycenter[obj0.x * h + obj0.y],
            barycenter[obj1.x * h + obj1.y]
          );
          set_pixel_object(object, obj1.x, obj1.y, h, obj0);
          barycenter[obj0.x * h + obj0.y] = new_barycenter;
        }
      }
    }
  }

}


__kernel void poiextraction(
  __global unsigned char* img, 
  __global short2* object, 
  __global float4* barycenter, 
  __global float4* results, 
  const int w, const int h, 
  const int sub_w, const int sub_h, 
  const int threshold, const int obj_per_partition)
{
  const int depth = 3;
  int work_id_x = get_global_id(0);
  int work_id_y = get_global_id(1);
  const int dim_x = depth * h;
  const int dim_y = depth;

  int start_x = work_id_x * sub_w;
  int start_y = work_id_y * sub_h;
  

  /** partition start index*/
  int pid = (work_id_x * get_global_size(1) + work_id_y) * obj_per_partition;
  /** cancelling unused objects */
  for (int oid = 0; oid < obj_per_partition; ++oid) {
    results[pid + oid].z = -1.0f;
  }
  /** object id */
  int oid = 0;
  for (int ox = start_x; ox < start_x + sub_w && oid < obj_per_partition; ++ ox) {
    for (int oy = start_y; oy < start_y + sub_h && oid < obj_per_partition; ++oy)
    {
      short2 pixel_obj = get_pixel_object(object, ox, oy, h);
      if (pixel_obj.x == ox && pixel_obj.y == oy && barycenter[ox * h + oy].z >= 50.0f) {
        results[pid + oid] = barycenter[ox * h + oy];
        oid++;
        short bx = barycenter[ox * h + oy].x;
        short by = barycenter[ox * h + oy].y;
        float weight =  barycenter[ox * h + oy].z;
        uchar weight_color = 255;// (uchar) min(max(weight,100.0f), 255.0f);
        int star_size = 20;
        for (int i = bx - star_size; i < bx + star_size; ++i) { 
          img[i *dim_x + by * dim_y + 2] = 0;
          img[i *dim_x + by * dim_y + 0] = 0;
          img[i *dim_x + by * dim_y + 1] = weight_color;
        }
        for (int j = by - star_size; j < by + star_size; ++j) {
          img[bx *dim_x + j * dim_y + 2] = 255;
          img[bx *dim_x + j * dim_y + 0] = 0;
          img[bx *dim_x + j * dim_y + 1] = 0;
        }
      }
    }
  }

}
