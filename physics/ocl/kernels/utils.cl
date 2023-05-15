// Most defines are in define.cl
// define.cl must be included as first file.cl to create OpenCL program

inline float genRandomNormalizedFloat(unsigned int i)
{
  // Linear Congruential Generator (LCG) parameters
  const ulong a = 1664525;
  const ulong c = 1013904223;
  const ulong m = 4294967296; // 2^32
    
  // Compute the random value using LCG
  ulong value = (a * i + c) % m;
    
  return (float)value / (float)m;
}

inline float3 genRandomNormalizedFloat3(unsigned int i)
{
  // Linear Congruential Generator (LCG) parameters
  const ulong a = 1664525;
  const ulong c = 1013904223;
  const ulong m = 4294967296; // 2^32
    
  // Compute the random value using LCG
  float x = (float)((a * i + c) % m) / (float) m;
  float y = (float)((a * i * i + c) % m) / (float) m;
  float z = (float)((a * i * i * i + c) % m) / (float) m;
    
  return (float3)(x,y,z);
}

/*
  Reset camera distance buffer
*/
__kernel void resetCameraDist(__global uint *cameraDist)
{
  cameraDist[ID] = (uint)(FAR_DIST);
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
  // Hack to be able to sort the cameraDist buffer using radix sort with closest particles coming last to be drawn on top using blending
  // We multiply squared length by 100 to have more precision before switching to uint
  cameraDist[ID] = (uint)(max(FAR_DIST - length(pos[ID].xyz - cameraPos[0].xyz) * 100.0f, 0.0f));
}

/*
  Fill position buffer with inf positions
*/
__kernel void infPosVerts(__global float4 *pos)
{
  pos[ID] = (float4)(FAR_DIST, FAR_DIST, FAR_DIST, 0.0f);
}

/*
  Fill color buffer with physical buffer for display and analysis
*/
__kernel void fillColorFloat(//Input
                             const  __global float  *physicalQuantity, // 0
                            //Param
                             const           float minVal, // 1
                             const           float maxVal, // 2
                            //Output
                                    __global float4 *col)  // 3
{
  float val = (physicalQuantity[ID] - minVal) / (maxVal - minVal);
  val *= step(0.0f, val);
  val *= step(val, 1.0f);
  //col[ID] = (float4)(val, 0.0f, 0.3f, 1.0f);
  col[ID] = (float4)(val, val, val, val);
}