#ifndef __OCLRADIXSORT_HPP
#define __OCLRADIXSORT_HPP

//
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include <algorithm>
#include <numeric>

#include <iostream>
#include <sstream>
#include <string_view>


template <size_t v>
struct is_power_of_2
{
  static constexpr bool value = v == 0 || ((v - 1) & v) == 0;
};

// cl type for integral type
template <typename T,
    typename = std::enable_if_t<
        std::is_unsigned<T>::value && std::is_integral<T>::value && sizeof(T) <= 8 && is_power_of_2<sizeof(T)>::value>>
struct cl_type_name
{
  inline static constexpr const char* name() noexcept
  {
    switch (sizeof(T))
    {
    case 1:
      return "uchar";
    case 2:
      return "ushort";
    case 4:
      return "uint";
    case 8:
      return "ulong";

    default:
      return "XXX bad type XXX";
    }
  }
};

// Copies buffer to device using non-default Command Queue
template <typename IteratorType>
cl_int toCL(const cl::CommandQueue& queue, IteratorType begin, IteratorType end,
    cl::Buffer& buffer)
{
  typedef typename std::iterator_traits<IteratorType>::value_type DataType;
  return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0,
      (end - begin) * sizeof(DataType), &(*begin));
}

// Copies buffer from device using non-default Command Queue
template <typename IteratorType>
cl_int fromCL(const cl::CommandQueue& queue, cl::Buffer& buffer,
    IteratorType begin, IteratorType end)
{
  typedef typename std::iterator_traits<IteratorType>::value_type DataType;
  return queue.enqueueReadBuffer(buffer, CL_TRUE, 0,
      (end - begin) * sizeof(DataType), &(*begin));
}

inline void swap(cl::Buffer& a, cl::Buffer& b)
{
  auto temp = a;
  a = b;
  b = temp;
}

template <typename T>
struct Opt
{
  const char* name;
  const T value;

  Opt(const char* name, T value)
      : name(name)
      , value(value)
  {
  }
};

namespace std
{
template <typename T>
ostream& operator<<(ostream& stream, const Opt<T>& opt)
{
  return stream << "-D" << opt.name << '=' << opt.value;
}

} // namespace std

template <typename... OT>
cl_int buildCLProgramWithOptions(const cl::Program& program, OT&&... args)
{
  std::ostringstream stream { " " };
  ((stream << args << ' '), ...);
  if (sizeof(void*) < 8)
  {
    stream << " -D HOST_PTR_IS_32bit ";
  }
  return program.build(stream.str().data());
}

using namespace std::literals::string_view_literals;

static constexpr std::string_view _src_other {
  R"CLC(
#ifdef HOST_PTR_IS_32bit
    #define SIZE uint
#else
    #define SIZE ulong
#endif

inline SIZE index(SIZE i, SIZE n) {
#ifdef TRANSPOSE
    const SIZE k = i / ( n / (_GROUPS * _ITEMS) );
    const SIZE l = i % ( n / (_GROUPS * _ITEMS) );
    return l * (_GROUPS * _ITEMS) + k;
#else
    return i;
#endif
}
)CLC"sv
};

// this kernel creates histograms from key vector
// defines required: _RADIX (radix), _BITS (size of radix in bits)
// it is possible to unroll 2 loops inside this kernel, take this into account
// when providing options to CL C compiler
static constexpr std::string_view _src_kernelHistogram {
  R"CLC(
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void histogram(
// in
    const global KEY_TYPE *keys,
    const SIZE length,
    const int pass,
// out
    global KEY_TYPE *global_histograms,
// local
    local  KEY_TYPE *histograms
) {
    const uint group  = get_group_id (0);
    const uint item   = get_local_id (0);
    const uint i_g    = get_global_id(0);

#if (__OPENCL_VERSION__ >= 200)
    __attribute__((opencl_unroll_hint(_RADIX)))
#endif
    for (int i = 0; i < _RADIX; ++i) { histograms[i * _ITEMS + item] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);

    const SIZE size = length / ( _GROUPS * _ITEMS );
    const SIZE start = i_g * size;

    for (SIZE i = start; i < start + size; ++i) {
        const KEY_TYPE key = keys[ index(i, length) ];
        const KEY_TYPE shortKey = ((key >> (pass * _BITS)) & (_RADIX - 1));
        ++histograms[shortKey * _ITEMS + item];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if (__OPENCL_VERSION__ >= 200)
    __attribute__((opencl_unroll_hint(_RADIX)))
#endif
    for (int i = 0; i < _RADIX; ++i) {
        global_histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item] = histograms[i * _ITEMS + item];
    }
}
)CLC"sv
};

// this kernel updates histograms with global sum after scan
static constexpr std::string_view _src_kernelMerge {
  R"CLC(
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void merge(
// in
    const global KEY_TYPE *sum,
// in-out
    global KEY_TYPE *histogram
) {
    const KEY_TYPE s = sum[ get_group_id(0) ];
    const uint gid2 = get_global_id(0) << 1;

    histogram[gid2]     += s;
    histogram[gid2 + 1] += s;
}
)CLC"sv
};

static constexpr std::string_view _src_kernelTranspose {
  R"CLC(
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void transpose(
// in
        const global KEY_TYPE *keysIn,
        const global INDEX_TYPE *permutationIn,
        const SIZE colCount,
        const SIZE rowCount,
// out
        global KEY_TYPE *keysOut,
        global INDEX_TYPE *permutationOut,
// local
        local KEY_TYPE *blockmat,
        local INDEX_TYPE *blockperm
) {
    const int i0      = get_global_id(0) * _TILESIZE;  // first row index
    const int j       = get_global_id(1);             // column index
    const int j_local = get_local_id(1);              // local column index

#if (__OPENCL_VERSION__ >= 200)
    __attribute__((opencl_unroll_hint(_TILESIZE)))
#endif
    for (int i = 0; i < _TILESIZE; ++i) {
        const int k = (i0 + i) * colCount + j;
        blockmat [i * _TILESIZE + j_local] = keysIn[k];
#ifdef COMPUTE_PERMUTATION
        blockperm[i * _TILESIZE + j_local] = permutationIn[k];
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    const int j0 = get_group_id(1) * _TILESIZE;
#if (__OPENCL_VERSION__ >= 200)
    __attribute__((opencl_unroll_hint(_TILESIZE)))
#endif
    for (int i = 0; i < _TILESIZE; ++i) {
        const int k = (j0 + i) * rowCount + i0 + j_local;
        keysOut[k]        = blockmat [j_local * _TILESIZE + i];
#ifdef COMPUTE_PERMUTATION
        permutationOut[k] = blockperm[j_local * _TILESIZE + i];
#endif
    }
}
)CLC"sv
};

// see Blelloch 1990
static constexpr std::string_view _src_kernelScan {
  R"CLC(
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void scan(
// in-out
    global KEY_TYPE *input,
// out
    global KEY_TYPE *sum,
// local
    local  KEY_TYPE *temp
) {
    const int gid2  = get_global_id(0) << 1;
    const int group = get_group_id(0);
    const int item  = get_local_id(0);
    const int n     = get_local_size(0) << 1;

    temp[2 * item]     = input[gid2];
    temp[2 * item + 1] = input[gid2 + 1];

    // parallel prefix sum (algorithm of Blelloch 1990)
    int decale = 1;
    // up sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (item < d) {
            const int ai = decale * ((item << 1) + 1) - 1;
            const int bi = decale * ((item << 1) + 2) - 1;
            temp[bi] += temp[ai];
        }
        decale <<= 1;
    }

    // store the last element in the global sum vector
    // (maybe used in the next step for constructing the global scan)
    // clear the last element
    if (item == 0) {
        sum[group] = temp[n - 1];
        temp[n - 1] = 0;
    }

    // down sweep phase
    for (int d = 1; d < n; d <<= 1) {
        decale >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (item < d) {
            const int ai = decale * ((item << 1) + 1) - 1;
            const int bi = decale * ((item << 1) + 2) - 1;
            const int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    input[gid2]     = temp[ item << 1 ];
    input[gid2 + 1] = temp[(item << 1) + 1];
}
)CLC"sv
};

static constexpr std::string_view _src_kernelReorder {
  R"CLC(
__attribute__((vec_type_hint(KEY_TYPE)))
kernel void reorder(
// in
    const global KEY_TYPE *keysIn,
    const global INDEX_TYPE *permutationIn,
    const SIZE length,
    const global KEY_TYPE *histograms,
    const int pass,
// out
    global KEY_TYPE *keysOut,
    global INDEX_TYPE *permutationOut,
// local
    local KEY_TYPE *local_histograms
) {
    const int item = get_local_id(0);
    const int group = get_group_id(0);

    const SIZE size = length / (_GROUPS * _ITEMS);
    const SIZE start = get_global_id(0) * size;

#if (__OPENCL_VERSION__ >= 200)
    __attribute__((opencl_unroll_hint(_RADIX)))
#endif
    for (int i = 0; i < _RADIX; ++i) {
        local_histograms[i * _ITEMS + item] = histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (SIZE i = start; i < start + size; ++i) {
        const KEY_TYPE key = keysIn[index(i, length)];
        const KEY_TYPE digit = ((key >> (pass * _BITS)) & (_RADIX - 1));
        const KEY_TYPE newPosition = local_histograms[digit * _ITEMS + item];

        local_histograms[digit * _ITEMS + item] = newPosition + 1;

        // WRITE TO GLOBAL (slow)
        keysOut       [index(newPosition, length)] = key;
#ifdef COMPUTE_PERMUTATION
        permutationOut[index(newPosition, length)] = permutationIn[index(i, length)];
#endif
        //
    }
}
)CLC"sv
};

static const cl::Program::Sources& _radixSortSources {
  { _src_other.data(), _src_other.length() },
  { _src_kernelHistogram.data(), _src_kernelHistogram.length() },
  { _src_kernelMerge.data(), _src_kernelMerge.length() },
  { _src_kernelReorder.data(), _src_kernelReorder.length() },
  { _src_kernelScan.data(), _src_kernelScan.length() },
  { _src_kernelTranspose.data(), _src_kernelTranspose.length() }
};

template <typename T,
    typename = std::enable_if_t<std::is_floating_point<T>::value>>
struct ProfilingInfo
{
  T transpose, histogram, scan, reorder;

  inline void reset()
  {
    transpose = static_cast<T>(0);
    scan = static_cast<T>(0);
    reorder = static_cast<T>(0);
    histogram = static_cast<T>(0);
  }

  inline constexpr T total() const
  {
    return transpose + scan + reorder + histogram;
  }
};

template <
    int bits, int totalBits,

    typename _DataType, typename _IndexType,

    size_t groups, // TODO move this to runtime?
    size_t items, // todo same^

    bool computePermutation,

    size_t histosplit,

    bool transpose, // todo same^
    size_t tileSize, // todo same^

    bool enableProfiling,

    typename ProfilingInfoType = double, int passes = totalBits / bits,

    _DataType radix = 1 << bits,
    _DataType maxInt = (static_cast<_DataType>(1) << (static_cast<_DataType>(totalBits) - 1)) - static_cast<_DataType>(1),

    typename = std::enable_if_t<
        std::is_integral<_DataType>::value && std::is_integral<_IndexType>::value && (totalBits / 8 <= sizeof(_DataType)) && is_power_of_2<groups>::value && is_power_of_2<items>::value && (totalBits % bits == 0) && ((groups * items * radix) % histosplit == 0)>>
class RadixSortBase
{
  using HistogramType = _DataType;

  const cl::Context ctx;
  const cl::Device device;

  cl::CommandQueue queue;
  cl::Program program;

  cl::Kernel kernelTranspose;
  cl::Kernel kernelHistogram;
  cl::Kernel kernelScan;
  cl::Kernel kernelMerge;
  cl::Kernel kernelReorder;

  cl::Buffer deviceHistograms;
  cl::Buffer deviceSum;
  cl::Buffer deviceTempSum;

  cl::Buffer keysIn;
  cl::Buffer keysOut;
  cl::Buffer permutationIn;
  cl::Buffer permutationOut;

  std::unique_ptr<std::vector<_IndexType>> _permutation;

  ProfilingInfo<ProfilingInfoType> _profilingInfo;

  // helper vars
  bool bound = false;

  // transpose params
  cl::NDRange localWorkItemsTranspose;
  size_t colCount;
  size_t rowCount;
  _IndexType tileSizeCalibrated;

  static constexpr inline ProfilingInfoType
  eventTiming(const cl::Event& event)
  {
    if (enableProfiling)
    {
      return static_cast<ProfilingInfoType>(
          event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>());
    }
    return static_cast<ProfilingInfoType>(0);
  }

  template <typename Iter>
  void _init(Iter begin, Iter end)
  {
    const size_t baseSize = end - begin;
    const size_t rest = baseSize % (groups * items);
    const size_t size = rest == 0 ? baseSize : (baseSize - rest + (groups * items));
    const size_t sizeInBytes = sizeof(_DataType) * size;

    std::cout << sizeof(_DataType) << "size data type";

    auto radixt = radix;
    auto maxInti = maxInt;

    /*
  std::cout<< "
   int bits, int totalBits,

    typename _DataType, typename _IndexType,

    size_t groups, // TODO move this to runtime?
    size_t items, // todo same^

    bool computePermutation,

    size_t histosplit,

    bool transpose, // todo same^
    size_t tileSize, // todo same^

    bool enableProfiling,

    typename ProfilingInfoType = double, int passes = totalBits / bits,

    _DataType radix = 1 << bits,
    _DataType maxInt = (static_cast<_DataType>(1) << (static_cast<_DataType>(totalBits) - 1)) - static_cast<_DataType>(1),


*/

    keysIn = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeInBytes };
    toCL(queue, begin, end, keysIn);
    if (rest != 0)
    {
      // TODO do not allocate vector? enqMap + std::fill maybe?
      std::vector<_DataType> pad(groups * items - rest, maxInt);
      queue.enqueueWriteBuffer(keysIn, CL_TRUE, sizeof(_DataType) * size,
          sizeof(_DataType) * pad.size(), pad.data());
    }
    keysOut = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeInBytes };
    queue.finish();

    deviceHistograms = cl::Buffer {
      ctx, CL_MEM_READ_WRITE, sizeof(HistogramType) * radix * groups * items
    };
    deviceSum = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeof(HistogramType) * histosplit };
    deviceTempSum = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeof(HistogramType) * histosplit };

    if (computePermutation)
    {
      _permutation = std::make_unique<std::vector<_IndexType>>(size);
      std::iota(_permutation->begin(), _permutation->end(), 0);
      permutationIn = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeof(_IndexType) * size };
      toCL(queue, _permutation->begin(), _permutation->end(), permutationIn);
      queue.finish();
      permutationOut = cl::Buffer { ctx, CL_MEM_READ_WRITE, sizeof(_IndexType) * size };
    }

    if (transpose)
    {
      rowCount = groups * items;
      colCount = size / rowCount;
      tileSizeCalibrated = (rowCount % tileSize != 0 || colCount % tileSize != 0) ? 1 : tileSize;
      kernelTranspose.setArg(4, tileSizeCalibrated);
      kernelTranspose.setArg(
          7, sizeof(HistogramType) * tileSizeCalibrated * tileSizeCalibrated,
          nullptr);
      kernelTranspose.setArg(
          8, sizeof(HistogramType) * tileSizeCalibrated * tileSizeCalibrated,
          nullptr);
      localWorkItemsTranspose = { 1, tileSizeCalibrated };
    }

    kernelScan.setArg(
        2,
        sizeof(HistogramType) * std::max(histosplit, radix * groups * items / histosplit),
        nullptr);

    kernelHistogram.setArg(1, size);
    kernelHistogram.setArg(3, deviceHistograms);
    kernelHistogram.setArg(4, sizeof(HistogramType) * radix * items, nullptr);

    kernelMerge.setArg(0, deviceSum);
    kernelMerge.setArg(1, deviceHistograms);

    kernelReorder.setArg(2, size);
    kernelReorder.setArg(3, deviceHistograms);
    kernelReorder.setArg(7, sizeof(HistogramType) * radix * items, nullptr);

    this->bound = true;
  }

  template <bool back>
  void _transpose()
  {
    kernelTranspose.setArg(0, keysIn);
    kernelTranspose.setArg(1, permutationIn);
    if (back)
    {
      kernelTranspose.setArg(2, rowCount);
      kernelTranspose.setArg(3, colCount);
    }
    else
    {
      kernelTranspose.setArg(2, colCount);
      kernelTranspose.setArg(3, rowCount);
    }
    kernelTranspose.setArg(5, keysOut);
    kernelTranspose.setArg(6, permutationOut);

    cl::NDRange globalWorkItemsTranspose = back ? cl::NDRange { colCount / tileSizeCalibrated, rowCount }
                                                : cl::NDRange { rowCount / tileSizeCalibrated, colCount };
    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelTranspose, { 0, 0 },
          globalWorkItemsTranspose,
          localWorkItemsTranspose, nullptr, &event);
      queue.finish();
      _profilingInfo.transpose += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelTranspose, { 0, 0 },
          globalWorkItemsTranspose,
          localWorkItemsTranspose);
      queue.finish();
    }

    swap(keysIn, keysOut);
    swap(permutationIn, permutationOut);
  }

  void _histogram(int pass)
  {
    kernelHistogram.setArg(0, keysIn);
    kernelHistogram.setArg(2, pass);

    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelHistogram, 0, groups * items, items,
          nullptr, &event);
      queue.finish();
      _profilingInfo.histogram += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelHistogram, 0, groups * items, items);
      queue.finish();
    }
  }

  void _scan()
  {
    kernelScan.setArg(0, deviceHistograms);
    kernelScan.setArg(1, deviceSum);

    size_t totalLocalScanItems = radix * groups * items / 2;
    size_t localItems = totalLocalScanItems / histosplit;
    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelScan, 0, totalLocalScanItems, localItems,
          nullptr, &event);
      queue.finish();
      _profilingInfo.scan += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelScan, 0, totalLocalScanItems,
          localItems);
      queue.finish();
    }

    kernelScan.setArg(0, deviceSum);
    kernelScan.setArg(1, deviceTempSum);

    totalLocalScanItems = histosplit / 2;
    localItems = totalLocalScanItems;
    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelScan, 0, totalLocalScanItems, localItems,
          nullptr, &event);
      queue.finish();
      _profilingInfo.scan += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelScan, 0, totalLocalScanItems,
          localItems);
      queue.finish();
    }

    totalLocalScanItems = radix * groups * items / 2;
    localItems = totalLocalScanItems / histosplit;
    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelMerge, 0, totalLocalScanItems,
          localItems, nullptr, &event);
      queue.finish();
      _profilingInfo.scan += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelMerge, 0, totalLocalScanItems,
          localItems);
      queue.finish();
    }
  }

  void _reorder(int pass)
  {
    kernelReorder.setArg(0, keysIn);
    kernelReorder.setArg(1, permutationIn);
    kernelReorder.setArg(4, pass);
    kernelReorder.setArg(5, keysOut);
    kernelReorder.setArg(6, permutationOut);

    if (enableProfiling)
    {
      cl::Event event;
      queue.enqueueNDRangeKernel(kernelReorder, 0, groups * items, items,
          nullptr, &event);
      queue.finish();
      _profilingInfo.reorder += eventTiming(event);
    }
    else
    {
      queue.enqueueNDRangeKernel(kernelReorder, 0, groups * items, items);
      queue.finish();
    }

    swap(keysIn, keysOut);
    swap(permutationIn, permutationOut);
  }

  template <typename Iter>
  void _sort(Iter begin, Iter end)
  {
    if (enableProfiling)
    {
      _profilingInfo.reset();
    }

    if (transpose)
    {
      _transpose<false>();
    }

    for (int pass = 0; pass < passes; ++pass)
    {
      _histogram(pass);
      _scan();
      _reorder(pass);
    }

    if (transpose)
    {
      _transpose<true>();
    }

    // get data back to host
    if (computePermutation)
    {
      fromCL(queue, permutationIn, _permutation->begin(), _permutation->end());
    }
    fromCL(queue, keysIn, begin, end);
  }

  cl_int _recompileProgram()
  {
    {
      // try to build program with new parameters
      cl_int err;
      if ((err = buildCLProgramWithOptions(
               program, "-w", Opt { "_RADIX", radix }, Opt { "_BITS", bits },
               Opt { "_GROUPS", groups }, Opt { "_ITEMS", items },
               Opt { "_TILESIZE", tileSize },
               Opt { "KEY_TYPE", cl_type_name<_DataType>::name() },
               Opt { "INDEX_TYPE", cl_type_name<_IndexType>::name() },
               transpose ? "-D TRANSPOSE" : "",
               computePermutation ? "-D COMPUTE_PERMUTATION" : ""))
          != CL_SUCCESS)
      {
        return err;
      }
    }
    // recreate kernels
    kernelHistogram = cl::Kernel { program, "histogram" };
    kernelScan = cl::Kernel { program, "scan" };
    kernelMerge = cl::Kernel { program, "merge" };
    kernelReorder = cl::Kernel { program, "reorder" };
    kernelTranspose = cl::Kernel { program, "transpose" };

    return CL_SUCCESS; // TODO what if exceptions are disabled and we fail to
        // create kernels?
  }

  public:
  typedef _DataType KeyType;
  typedef _IndexType IndexType;

  /**
   * @brief Construct Radix sort
   * @param device device
   */
  explicit RadixSortBase(cl::Device device)
      : RadixSortBase(cl::Context { { device } }, device)
  {
  }

  /**
   * @brief Construct Radix sort
   * @param ctx context
   * @param device device
   */
  RadixSortBase(cl::Context ctx, cl::Device device)
      : ctx(ctx)
      , device(device)
      , queue(cl::CommandQueue {
            ctx, device, enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0 })
      , program(cl::Program { ctx, _radixSortSources })
  {
    // perform initial program compilation
    _recompileProgram();
  }

  /**
   * @brief Sort given iterable using radix sort algorithm.
   * @tparam rebind recreate permutation and all information needed to run sort
   * <p/>
   * @tparam Iter random access iterator
   * @param begin begin
   * @param end end
   */
  template <bool rebind = true, typename Iter>
  void sort(Iter begin, Iter end)
  {
    if (!bound || rebind)
    {
      _init(begin, end);
    }
    _sort(begin, end);
  }

  /**
   * @brief Retrieve profiling info from the last call to sort()
   * @return profiling info
   */
  inline const ProfilingInfo<ProfilingInfoType>&
  profilingInfo() const noexcept
  {
    return _profilingInfo;
  }

  /**
   * @brief Retrieve permutation vector built in the last call to sort()
   * @return permutation vector
   */
  inline typename std::enable_if<computePermutation,
      const std::vector<IndexType>&>::type
  permutation() const
  {
    return *_permutation;
  }

  /**
   * @brief Max possible value of vector element for this radix sort instance
   * @return max value
   */
  inline constexpr KeyType maxValue() const noexcept { return maxInt; }
};

/**
 * @brief Radix sort implementation
 * @tparam bits size of radix in bits
 * @tparam totalBits size of key in bits
 * @tparam DataType data type for key
 * @tparam IndexType data type for permutation vector
 * @tparam groups global work items count for algorithm
 * @tparam items local work items count for algorithm
 * @tparam computePermutation whether or not to compute permutation of key
 * vector.
 * @tparam histosplit size of histogram local part for "scan" part of Radix Sort
 * @tparam transpose whether or not to perform transposition of key vector. This
 * may improve cache usage on some plaforms
 * @tparam tileSize size of tile for transpose step
 * @tparam enableProfiling whether or not to enable commands profiling. Only
 * kernel executions are profiled!
 */
template <int bits = 8, int totalBits = 32,

    typename DataType = unsigned int, typename IndexType = unsigned int,

    size_t groups = 128, size_t items = 8,

    bool computePermutation = true,

    size_t histosplit = 512,

    bool transpose = true, size_t tileSize = 32,

    bool enableProfiling = false>
using RadixSort = RadixSortBase<bits, totalBits, DataType, IndexType, groups,
    items, computePermutation, histosplit,
    transpose, tileSize, enableProfiling>;

#endif