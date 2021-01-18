#include "RadixSort.hpp"

#include <ctime>
#include <iostream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

using namespace Core;

#define PROGRAM_RADIXSORT "RadixSort"

#define KERNEL_HISTOGRAM "histogram"
#define KERNEL_MERGE "merge"
#define KERNEL_SCAN "scan"
#define KERNEL_REORDER "reorder"

RadixSort::RadixSort(size_t numEntities)
    : m_numEntities(numEntities)
    , m_numRadix(256)
    , m_numRadixBits(8)
    , m_numTotalBits(32)
    , m_numGroups(128)
    , m_numItems(4)
    , m_histoSplit(256)
{
  m_indices.resize(m_numEntities);
  std::iota(m_indices.begin(), m_indices.end(), 0);

  m_numRadixPasses = m_numTotalBits / m_numRadixBits;

  createProgram();

  createBuffers();

  createKernels();
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

  // WIP, hardcoded Path
  clContext.createProgram(PROGRAM_RADIXSORT,
      "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\radixSort.cl",
      clBuildOptions.str());

  return true;
}

bool RadixSort::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  const size_t rest = m_numEntities % (m_numGroups * m_numItems);
  const size_t size = (rest == 0) ? m_numEntities : (m_numEntities - rest + (m_numGroups * m_numItems));
  const size_t sizeInBytes = sizeof(unsigned int) * size;

  unsigned int maxInt = (static_cast<unsigned int>(1) << (static_cast<unsigned int>(m_numTotalBits) - 1)) - static_cast<unsigned int>(1);
  //unsigned short maxInt = (static_cast<unsigned short>(1) << (static_cast<unsigned short>(m_numTotalBits) - 1)) - static_cast<unsigned short>(1);
  //unsigned int maxInt = 9;

  std::vector<unsigned int> keys(m_numEntities, 0);
  auto rng = Core::makeRng(maxInt);
  std::generate(keys.begin(), keys.end(), rng);
  //std::iota(keys.begin(), keys.end(), 0);
  //std::cout << "generating " << keys.size() << " keys...." << std::endl;

  clContext.createBuffer("RadixSortKeysIn", sizeInBytes, CL_MEM_READ_WRITE);
  clContext.loadBufferFromHost("RadixSortKeysIn", 0, sizeInBytes, keys.data());
  if (rest != 0)
  {
    std::vector<unsigned int> pad(m_numGroups * m_numItems - rest, 10000000);
    clContext.loadBufferFromHost("RadixSortKeysIn", sizeof(unsigned int) * size, sizeof(unsigned int) * pad.size(), pad.data());
  }
  clContext.createBuffer("RadixSortKeysOut", sizeInBytes, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortHistogram", sizeof(unsigned int) * m_numRadix * m_numGroups * m_numItems, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortSum", sizeof(unsigned int) * m_histoSplit, CL_MEM_READ_WRITE);
  clContext.createBuffer("RadixSortTempSum", sizeof(unsigned int) * m_histoSplit, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortIndicesIn", sizeof(unsigned int) * size, CL_MEM_READ_WRITE);
  clContext.loadBufferFromHost("RadixSortIndicesIn", 0, sizeof(unsigned int) * m_indices.size(), m_indices.data());

  clContext.createBuffer("RadixSortIndicesOut", sizeof(unsigned int) * size, CL_MEM_READ_WRITE);
  return true;
}

bool RadixSort::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  const size_t rest = m_numEntities % (m_numGroups * m_numItems);
  const size_t size = (rest == 0) ? m_numEntities : (m_numEntities - rest + (m_numGroups * m_numItems));

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_HISTOGRAM, { "RadixSortKeysIn", "", "", "RadixSortHistogram" });
  clContext.setKernelArg(KERNEL_HISTOGRAM, 1, sizeof(size), &size);
  clContext.setKernelArg(KERNEL_HISTOGRAM, 4, sizeof(unsigned int) * m_numRadix * m_numItems, nullptr);

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_SCAN, { "RadixSortHistogram", "RadixSortSum" });
  clContext.setKernelArg(KERNEL_SCAN, 2, sizeof(unsigned int) * std::max(m_histoSplit, m_numRadix * m_numGroups * m_numItems / m_histoSplit), nullptr);

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_MERGE, { "RadixSortSum", "RadixSortHistogram" });

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_REORDER, { "RadixSortKeysIn", "RadixSortIndicesIn", "", "RadixSortHistogram", "", "RadixSortKeysOut", "RadixSortIndicesOut" });
  clContext.setKernelArg(KERNEL_REORDER, 2, sizeof(size), &size);
  clContext.setKernelArg(KERNEL_REORDER, 7, sizeof(unsigned int) * m_numRadix * m_numItems, nullptr);

  return true;
}

void RadixSort::sort()
{
  CL::Context& clContext = CL::Context::Get();

  size_t totalScan = m_numRadix * m_numGroups * m_numItems / 2;
  size_t localScan = totalScan / m_histoSplit;

  for (int radixPass = 0; radixPass < m_numRadixPasses; ++radixPass)
  {
    clContext.setKernelArg(KERNEL_HISTOGRAM, 0, "RadixSortKeysIn");
    clContext.setKernelArg(KERNEL_HISTOGRAM, 2, sizeof(radixPass), &radixPass);
    clContext.runKernel(KERNEL_HISTOGRAM, m_numGroups * m_numItems, m_numItems);

    clContext.setKernelArg(KERNEL_SCAN, 0, "RadixSortHistogram");
    clContext.setKernelArg(KERNEL_SCAN, 1, "RadixSortSum");
    clContext.runKernel(KERNEL_SCAN, totalScan, localScan);

    clContext.setKernelArg(KERNEL_SCAN, 0, "RadixSortSum");
    clContext.setKernelArg(KERNEL_SCAN, 1, "RadixSortTempSum");
    clContext.runKernel(KERNEL_SCAN, m_histoSplit / 2, m_histoSplit / 2);

    clContext.runKernel(KERNEL_MERGE, totalScan, localScan);

    clContext.setKernelArg(KERNEL_REORDER, 0, "RadixSortKeysIn");
    clContext.setKernelArg(KERNEL_REORDER, 1, "RadixSortIndicesIn");
    clContext.setKernelArg(KERNEL_REORDER, 5, "RadixSortKeysOut");
    clContext.setKernelArg(KERNEL_REORDER, 6, "RadixSortIndicesOut");
    clContext.setKernelArg(KERNEL_REORDER, 4, sizeof(radixPass), &radixPass);
    clContext.runKernel(KERNEL_REORDER, m_numGroups * m_numItems, m_numItems);

    clContext.swapBuffers("RadixSortKeysIn", "RadixSortKeysOut");
    clContext.swapBuffers("RadixSortIndicesIn", "RadixSortIndicesOut");
  }

  std::vector<unsigned int> keysIn(m_numEntities, 0);
  clContext.unloadBufferFromDevice("RadixSortKeysIn", 0, sizeof(unsigned int) * keysIn.size(), keysIn.data());
  /*
  std::vector<unsigned int> indicesIn(m_numEntities, 0);
  clContext.unloadBufferFromDevice("RadixSortIndicesIn", 0, sizeof(unsigned int) * indicesIn.size(), indicesIn.data());

  std::vector<unsigned int> keysOut(m_numEntities, 0);
  clContext.unloadBufferFromDevice("RadixSortKeysOut", 0, sizeof(unsigned int) * keysOut.size(), keysOut.data());

  std::vector<unsigned int> indicesOut(m_numEntities, 0);
  clContext.unloadBufferFromDevice("RadixSortIndicesOut", 0, sizeof(unsigned int) * indicesOut.size(), indicesOut.data());
*/
  std::cout << "Sorted: " << std::boolalpha << std::is_sorted(keysIn.begin(), keysIn.end()) << std::endl;
}