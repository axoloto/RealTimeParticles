
__kernel void matvec_mult(__global float4 *matrix, __global float4 *vector, __global float *result)
{
  int i = get_global_id(0);
  result[i] = dot(matrix[i], vector[0]);
}

__kernel void randPosVerts(__global float4 *randomPos)
{
  int i = get_global_id(0);
  randomPos[i] = (float4) (i, i, i, i);
}

__kernel void colorVerts(__global float4 *color)
{
  int i = get_global_id(0);
  int all = get_global_size(0);
  float col = i / all;
  color[i] = (float4) (col, 1.0 - col, col, 1.0);
}