// Preprocessor defines following constant variables in RadixSort.cpp
// _RADIX            - number of radix
// _BITS             - size of radix in bits
// _GROUPS           - number of work groups
// _ITEMS            - number of work items
// HOST_PTR_IS_32bit - only if 32bit OS

#ifdef HOST_PTR_IS_32bit
#define SIZE uint
#else
#define SIZE ulong
#endif

#define ID get_global_id(0)

/*
  Create histograms from key vector
*/
__kernel void histogram(//Input
                        const __global uint *keys,              // 0
                        const          SIZE length,             // 1
                        const          int  pass,               // 2
                        //Output
                              __global uint *global_histograms, // 3
                        //Local
                              __local  uint *histograms)        // 4
    
{
  const uint group = get_group_id(0);
  const uint item = get_local_id(0);
  const uint i_g = get_global_id(0);

  for (int i = 0; i < _RADIX; ++i)
  {
    histograms[i * _ITEMS + item] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const SIZE size = length / (_GROUPS * _ITEMS);
  const SIZE start = i_g * size;

  for (SIZE i = start; i < start + size; ++i)
  {
    const uint key = keys[i];
    const uint shortKey = ((key >> (pass * _BITS)) & (_RADIX - 1));
    ++histograms[shortKey * _ITEMS + item];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < _RADIX; ++i)
  {
    global_histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item] = histograms[i * _ITEMS + item];
  }
}

/*
  Update histograms with global sum after scan
*/
__kernel void merge(//Input
                    const __global uint *sum,       // 0
                    //Input/Output
                          __global uint *histogram) // 1

{
  const uint s = sum[get_group_id(0)];
  const uint gid2 = get_global_id(0) << 1;

  histogram[gid2] += s;
  histogram[gid2 + 1] += s;
}

// see Blelloch 1990
__kernel void scan(//Input/Output
                   __global uint *input, // 0
                   //Output
                   __global uint *sum,   // 1
                   //Local
                   __local  uint *temp)  // 2
{
  const int gid2 = get_global_id(0) << 1;
  const int group = get_group_id(0);
  const int item = get_local_id(0);
  const int n = get_local_size(0) << 1;

  temp[2 * item] = input[gid2];
  temp[2 * item + 1] = input[gid2 + 1];

  // parallel prefix sum (algorithm of Blelloch 1990)
  int decale = 1;
  // up sweep phase
  for (int d = n >> 1; d > 0; d >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item < d)
    {
      const int ai = decale * ((item << 1) + 1) - 1;
      const int bi = decale * ((item << 1) + 2) - 1;
      temp[bi] += temp[ai];
    }
    decale <<= 1;
  }

  // store the last element in the global sum vector
  // (maybe used in the next step for constructing the global scan)
  // clear the last element
  if (item == 0)
  {
    sum[group] = temp[n - 1];
    temp[n - 1] = 0;
  }

  // down sweep phase
  for (int d = 1; d < n; d <<= 1)
  {
    decale >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item < d)
    {
      const int ai = decale * ((item << 1) + 1) - 1;
      const int bi = decale * ((item << 1) + 2) - 1;
      const int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  input[gid2] = temp[item << 1];
  input[gid2 + 1] = temp[(item << 1) + 1];
}

__kernel void reorder(//Input
                      const __global uint *keysIn,           // 0
                      const __global uint *permutationIn,    // 1
                      const          SIZE length,            // 2
                      const __global uint *histograms,       // 3
                      const          int  pass,              // 4
                      //Output
                            __global uint *keysOut,          // 5 
                            __global uint *permutationOut,   // 6
                      //Local
                            __local  uint *local_histograms) // 7
{
  const int item = get_local_id(0);
  const int group = get_group_id(0);

  const SIZE size = length / (_GROUPS * _ITEMS);
  const SIZE start = get_global_id(0) * size;

  for (int i = 0; i < _RADIX; ++i)
  {
    local_histograms[i * _ITEMS + item] = histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (SIZE i = start; i < start + size; ++i)
  {
    const uint key = keysIn[i];
    const uint digit = ((key >> (pass * _BITS)) & (_RADIX - 1));
    const uint newPosition = local_histograms[digit * _ITEMS + item];

    local_histograms[digit * _ITEMS + item] = newPosition + 1;

    // WRITE TO GLOBAL (slow)
    keysOut[newPosition] = key;
    permutationOut[newPosition] = permutationIn[i];
    //
  }
}

__kernel void resetIndex(__global uint* indices)
{
  indices[ID] = ID;
}

/*
  Permutate float4 values.
*/
__kernel void permutate(//Input
                        const __global uint   *permutatedIndices, // 0
                        const __global float4 *valToPermutate,    // 1
                        //Output
                              __global float4 *permutatedVal)     // 2
{
  const uint newIndex = permutatedIndices[ID];

  permutatedVal[ID] = valToPermutate[newIndex];
}

/*
  Permutate uint values.
*/
__kernel void permutateInt(//Input
                           const __global uint *permutatedIndices, // 0
                           const __global uint *valToPermutate,    // 1
                           //Output
                                 __global uint *permutatedVal)     // 2
{
  const uint newIndex = permutatedIndices[ID];

  permutatedVal[ID] = valToPermutate[newIndex];
}