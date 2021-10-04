
#define FAR_DIST      100000000.0f
#define ID            get_global_id(0)

/*
  Random unsigned integer number generator
*/
inline unsigned int parallelRNG(unsigned int i)
{
  unsigned int value = i;

  value = (value ^ 61) ^ (value >> 16);
  value *= 9;
  value ^= value << 4;
  value *= 0x27d4eb2d;
  value ^= value >> 15;

  return value;
}

/*
  Reset camera distance buffer
*/
__kernel void resetCameraDist(__global uint *cameraDist)
{
  cameraDist[ID] = (uint)(FAR_DIST * 2);
}

/*
  Fill camera distance buffer
*/
__kernel void fillCameraDist(//Input
                             const __global float4 *pos,          // 0
                             const __global float3 *cameraPos,    // 1
                             //Output
                                   __global uint   *cameraDist)   // 2
{
  cameraDist[ID] = (uint)(FAR_DIST - fast_length(pos[ID].xyz - cameraPos[0].xyz));
}

/*
  Fill position buffer with inf positions
*/
__kernel void infPosVerts(__global float4 *pos)
{
  pos[ID] = (float4)(FAR_DIST, FAR_DIST, FAR_DIST, 0.0f);
}