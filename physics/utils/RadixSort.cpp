#include "RadixSort.hpp"

#include "Logging.hpp"
#include <ctime>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace Physics;

#define PROGRAM_RADIXSORT "RadixSort"

#define KERNEL_RESET_INDEX "resetIndex"
#define KERNEL_HISTOGRAM "histogram"
#define KERNEL_MERGE "merge"
#define KERNEL_SCAN "scan"
#define KERNEL_REORDER "reorder"
#define KERNEL_PERMUTATE "permutate"

RadixSort::RadixSort(size_t numEntities)
    : m_numEntities(numEntities)
    , m_numRadix(256)
    , m_numRadixBits(8)
    , m_numTotalBits(32)
    , m_numGroups(128)
    , m_numItems(4)
    , m_histoSplit(256)
{
  m_numRadixPasses = m_numTotalBits / m_numRadixBits;

  if (m_numEntities % (m_numGroups * m_numItems) != 0)
    LOG_ERROR("Radix sort not supporting arrays of size {}, only ones whose size is multiple of {} ", m_numEntities, m_numGroups * m_numItems);

  if (!createProgram())
  {
    LOG_ERROR("Failed to initialize radix sort program");
    return;
  }

  if (!createBuffers())
  {
    LOG_ERROR("Failed to initialize radix sort buffers");
    return;
  }

  if (!createKernels())
  {
    LOG_ERROR("Failed to initialize radix sort kernels");
    return;
  }

  LOG_INFO("Radix sort correctly initialized");
}

bool RadixSort::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  std::ostringstream clBuildOptions;
  clBuildOptions << " -D_RADIX=" << m_numRadix;
  clBuildOptions << " -D_BITS=" << m_numRadixBits;
  clBuildOptions << " -D_GROUPS=" << m_numGroups;
  clBuildOptions << " -D_ITEMS=" << m_numItems;
  if (sizeof(void*) < 8)
  {
    clBuildOptions << " -DHOST_PTR_IS_32bit";
  }

  if (!clContext.createProgram(PROGRAM_RADIXSORT, "radixSort.cl", clBuildOptions.str()))
    return false;

  return true;
}

bool RadixSort::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createBuffer("RadixSortKeysTemp", sizeof(unsigned int) * m_numEntities, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortHistogram", sizeof(unsigned int) * m_numRadix * m_numGroups * m_numItems, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortSum", sizeof(unsigned int) * m_histoSplit, CL_MEM_READ_WRITE);
  clContext.createBuffer("RadixSortTempSum", sizeof(unsigned int) * m_histoSplit, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortIndices", sizeof(unsigned int) * m_numEntities, CL_MEM_READ_WRITE);
  clContext.createBuffer("RadixSortIndicesTemp", sizeof(unsigned int) * m_numEntities, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortPermutateTemp", 4 * sizeof(float) * m_numEntities, CL_MEM_READ_WRITE);

  return true;
}

bool RadixSort::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_RESET_INDEX, { "RadixSortIndices" });

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_HISTOGRAM, { "", "", "", "RadixSortHistogram" });
  clContext.setKernelArg(KERNEL_HISTOGRAM, 1, sizeof(size_t), &m_numEntities);
  clContext.setKernelArg(KERNEL_HISTOGRAM, 4, sizeof(unsigned int) * m_numRadix * m_numItems, nullptr);

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_SCAN, { "RadixSortHistogram", "RadixSortSum" });
  clContext.setKernelArg(KERNEL_SCAN, 2, sizeof(unsigned int) * std::max(m_histoSplit, m_numRadix * m_numGroups * m_numItems / m_histoSplit), nullptr);

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_MERGE, { "RadixSortSum", "RadixSortHistogram" });

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_REORDER, { "", "RadixSortIndices", "", "RadixSortHistogram", "", "RadixSortKeysTemp", "RadixSortIndicesTemp" });
  clContext.setKernelArg(KERNEL_REORDER, 2, sizeof(size_t), &m_numEntities);
  clContext.setKernelArg(KERNEL_REORDER, 7, sizeof(unsigned int) * m_numRadix * m_numItems, nullptr);

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_PERMUTATE, { "RadixSortIndices" });

  return true;
}

void RadixSort::sort(const std::string& inputKeyBufferName, const std::vector<std::string>& optionalInputBufferNames)
{
  // First sorting main input key buffer
  // Then sorting optional input buffers based on indices permutation of the main input key buffer

  CL::Context& clContext = CL::Context::Get();

  size_t totalScan = m_numRadix * m_numGroups * m_numItems / 2;
  size_t localScan = totalScan / m_histoSplit;

  clContext.runKernel(KERNEL_RESET_INDEX, m_numEntities);

  for (int radixPass = 0; radixPass < m_numRadixPasses; ++radixPass)
  {
    clContext.setKernelArg(KERNEL_HISTOGRAM, 0, inputKeyBufferName);
    clContext.setKernelArg(KERNEL_HISTOGRAM, 2, sizeof(radixPass), &radixPass);
    clContext.runKernel(KERNEL_HISTOGRAM, m_numGroups * m_numItems, m_numItems);

    clContext.setKernelArg(KERNEL_SCAN, 0, "RadixSortHistogram");
    clContext.setKernelArg(KERNEL_SCAN, 1, "RadixSortSum");
    clContext.runKernel(KERNEL_SCAN, totalScan, localScan);

    clContext.setKernelArg(KERNEL_SCAN, 0, "RadixSortSum");
    clContext.setKernelArg(KERNEL_SCAN, 1, "RadixSortTempSum");
    clContext.runKernel(KERNEL_SCAN, m_histoSplit / 2, m_histoSplit / 2);

    clContext.runKernel(KERNEL_MERGE, totalScan, localScan);

    clContext.setKernelArg(KERNEL_REORDER, 0, inputKeyBufferName);
    clContext.setKernelArg(KERNEL_REORDER, 1, "RadixSortIndices");
    clContext.setKernelArg(KERNEL_REORDER, 5, "RadixSortKeysTemp");
    clContext.setKernelArg(KERNEL_REORDER, 6, "RadixSortIndicesTemp");
    clContext.setKernelArg(KERNEL_REORDER, 4, sizeof(radixPass), &radixPass);
    clContext.runKernel(KERNEL_REORDER, m_numGroups * m_numItems, m_numItems);

    clContext.swapBuffers(inputKeyBufferName, "RadixSortKeysTemp");
    clContext.swapBuffers("RadixSortIndices", "RadixSortIndicesTemp");
  }

  for (const auto& bufferToPermutate : optionalInputBufferNames)
  {
    clContext.copyBuffer(bufferToPermutate, "RadixSortPermutateTemp");
    clContext.setKernelArg(KERNEL_PERMUTATE, 1, "RadixSortPermutateTemp");
    clContext.setKernelArg(KERNEL_PERMUTATE, 2, bufferToPermutate);
    clContext.runKernel(KERNEL_PERMUTATE, m_numEntities);
  }
}