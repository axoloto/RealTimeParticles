
#ifdef HOST_PTR_IS_32bit
#define SIZE uint
#else
#define SIZE ulong
#endif

inline SIZE index(SIZE i, SIZE n)
{
#ifdef TRANSPOSE
  const SIZE k = i / (n / (_GROUPS * _ITEMS));
  const SIZE l = i % (n / (_GROUPS * _ITEMS));
  return l * (_GROUPS * _ITEMS) + k;
#else
  return i;
#endif
}

// this kernel creates histograms from key vector
// defines required: _RADIX (radix), _BITS (size of radix in bits)
// it is possible to unroll 2 loops inside this kernel, take this into account
// when providing options to CL C compiler
__kernel void histogram(
    // in
    const global unsigned int* keys,
    const SIZE length,
    const int pass,
    // out
    global unsigned int* global_histograms,
    // local
    local unsigned int* histograms)
{
  const uint group = get_group_id(0);
  const uint item = get_local_id(0);
  const uint i_g = get_global_id(0);

#if (__OPENCL_VERSION__ >= 200)
  __attribute__((opencl_unroll_hint(_RADIX)))
#endif
  for (int i = 0; i < _RADIX; ++i)
  {
    histograms[i * _ITEMS + item] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const SIZE size = length / (_GROUPS * _ITEMS);
  const SIZE start = i_g * size;

  for (SIZE i = start; i < start + size; ++i)
  {
    const unsigned int key = keys[index(i, length)];
    const unsigned int shortKey = ((key >> (pass * _BITS)) & (_RADIX - 1));
    ++histograms[shortKey * _ITEMS + item];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if (__OPENCL_VERSION__ >= 200)
  __attribute__((opencl_unroll_hint(_RADIX)))
#endif
  for (int i = 0; i < _RADIX; ++i)
  {
    global_histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item] = histograms[i * _ITEMS + item];
  }
}

// this kernel updates histograms with global sum after scan
__kernel void merge(
    // in
    const global unsigned int* sum,
    // in-out
    global unsigned int* histogram)
{
  const unsigned int s = sum[get_group_id(0)];
  const uint gid2 = get_global_id(0) << 1;

  histogram[gid2] += s;
  histogram[gid2 + 1] += s;
}

// see Blelloch 1990
__kernel void scan(
    // in-out
    global unsigned int* input,
    // out
    global unsigned int* sum,
    // local
    local unsigned int* temp)
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
    const global unsigned int* keysIn,
    const global unsigned int* permutationIn,
    const SIZE length,
    const global unsigned int* histograms,
    const int pass,
    // out
    global unsigned int* keysOut,
    global unsigned int* permutationOut,
    // local
    local unsigned int* local_histograms)
{
  const int item = get_local_id(0);
  const int group = get_group_id(0);

  const SIZE size = length / (_GROUPS * _ITEMS);
  const SIZE start = get_global_id(0) * size;

#if (__OPENCL_VERSION__ >= 200)
  __attribute__((opencl_unroll_hint(_RADIX)))
#endif
  for (int i = 0; i < _RADIX; ++i)
  {
    local_histograms[i * _ITEMS + item] = histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (SIZE i = start; i < start + size; ++i)
  {
    const unsigned int key = keysIn[index(i, length)];
    const unsigned int digit = ((key >> (pass * _BITS)) & (_RADIX - 1));
    const unsigned int newPosition = local_histograms[digit * _ITEMS + item];

    local_histograms[digit * _ITEMS + item] = newPosition + 1;

    // WRITE TO GLOBAL (slow)
    keysOut[index(newPosition, length)] = key;
#ifdef COMPUTE_PERMUTATION
    permutationOut[index(newPosition, length)] = permutationIn[index(i, length)];
#endif
    //
  }
}