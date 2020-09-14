
__kernel void matvec_mult(__global float4 *matrix, __global float4 *vector, __global float *result)
{
  int i = get_global_id(0);
  result[i] = dot(matrix[i], vector[0]);
}

unsigned int parallelRNG( unsigned int i )
{
	unsigned int value = i;

	value = (value ^ 61) ^ (value>>16);
	value *= 9;
	value ^= value << 4;
	value *= 0x27d4eb2d;
	value ^= value >> 15;

	return value;
}

__kernel void randPosVerts(__global float4 *randomPos)
{
  unsigned int i = get_global_id(0);

  unsigned int randomIntX = parallelRNG(i);
  unsigned int randomIntY = parallelRNG(i + 1);
  unsigned int randomIntZ = parallelRNG(i + 2);

  float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0;
  float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0;
  float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0;

  float3 randomCoords = (float3) (x, y, z);

  randomPos[i].xyz = clamp(randomCoords, (float)-250.0, (float)250.0);
  randomPos[i].w = 1.0;
}

__kernel void colorVerts(__global float4 *color)
{
  int i = get_global_id(0);
  float col = i / (float) get_global_size(0);
  color[i] = (float4) (col, col, col, 1.0);
}