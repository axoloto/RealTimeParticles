
#ifdef HOST_PTR_IS_32bit
#define SIZE uint
#else
#define SIZE ulong
#endif

// this kernel creates histograms from key vector
// defines required: _RADIX (radix), _BITS (size of radix in bits)
// it is possible to unroll 2 loops inside this kernel, take this into account
// when providing options to CL C compiler
__kernel void histogram(
    // in
    const global uint* keys,
    const SIZE length,
    const int pass,
    // out
    global uint* global_histograms,
    // local
    local uint* histograms)
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

// this kernel updates histograms with global sum after scan
__kernel void merge(
    // in
    const global uint* sum,
    // in-out
    global uint* histogram)
{
  const uint s = sum[get_group_id(0)];
  const uint gid2 = get_global_id(0) << 1;

  histogram[gid2] += s;
  histogram[gid2 + 1] += s;
}

// see Blelloch 1990
__kernel void scan(
    // in-out
    global uint* input,
    // out
    global uint* sum,
    // local
    local uint* temp)
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

__kernel void reorder(
    // in
    const global uint* keysIn,
    const global uint* permutationIn,
    const SIZE length,
    const global uint* histograms,
    const int pass,
    // out
    global uint* keysOut,
    global uint* permutationOut,
    // local
    local uint* local_histograms)
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