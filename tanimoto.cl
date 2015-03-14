__kernel void and_or_func(__global const int *a_g, 
                          __global const int *b_g, 
                          __global int *and_g,
                          __global int *or_g) // hehe, orgy
{
  int gid = get_global_id(0);
  and_g[gid] = a_g[gid] & b_g[gid];
  or_g[gid] = a_g[gid] | b_g[gid];
}

__kernel void reduction(__global int *res_g, 
                        __global int *out, int stride)
{
  int gid = get_global_id(0);
  int gid2 = gid*stride;
  res_g[gid2] = res_g[gid2] + res_g[gid2+stride/2];
  out[0] = res_g[0];
}
