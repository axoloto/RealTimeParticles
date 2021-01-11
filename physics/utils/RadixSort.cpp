#include "RadixSort.hpp"

#include <ctime>
#include <iostream>
#include <numeric>
#include <spdlog/spdlog.h>

using namespace Core;

#define PROGRAM_RADIXSORT "RadixSort"

#define KERNEL_HISTOGRAM "histogram"
#define KERNEL_MERGE "merge"
#define KERNEL_SCAN "scan"
#define KERNEL_REORDER "reorder"

RadixSort::RadixSort(size_t numEntities)
    : m_numEntities(numEntities)
    , m_numRadix(128)
    , m_numRadixBits(8)
    , m_numTotalBits(32)
    , m_numGroups(128)
    , m_numItems(4)
{
  m_indices.resize(NUM_MAX_ENTITIES);
  std::iota(m_indices.begin(), m_indices.end(), 0);

  createProgram();

  createBuffers();

  createKernels();
}

bool RadixSort::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  std::string buildOptions = "-D_RADIX=128 -D_BITS=8 -D_GROUPS=128 -D_ITEMS=4 -D_TILESIZE=32";
  buildOptions += (sizeof(void*) < 8) ? " -DHOST_PTR_IS_32bit" : "";

  // WIP, hardcoded Path
  clContext.createProgram(PROGRAM_RADIXSORT,
      "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\radixSort.cl",
      buildOptions);

  return true;
}

bool RadixSort::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  const size_t rest = NUM_MAX_ENTITIES % (m_numGroups * m_numItems);
  const size_t size = (rest == 0) ? NUM_MAX_ENTITIES : (NUM_MAX_ENTITIES - rest + (m_numGroups * m_numItems));
  const size_t sizeInBytes = sizeof(unsigned int) * size;

  std::vector<unsigned int> keys(NUM_MAX_ENTITIES, 0);
  clContext.createBuffer("RadixSortKeysIn", sizeInBytes, CL_MEM_READ_WRITE);
  clContext.fillBuffer("RadixSortKeysIn", 0, sizeInBytes, keys.data());
  if (rest != 0)
  {
    std::vector<unsigned int> pad(m_numGroups * m_numItems - rest, 10000000);
    //clContext.fillBuffer("RadixSortKeysIn", sizeof(unsigned int) * size, sizeof(unsigned int) * pad.size(), pad.data());
  }
  clContext.createBuffer("RadixSortKeysOut", sizeInBytes, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortHistogram", sizeof(unsigned int) * m_numRadix * m_numGroups * m_numItems, CL_MEM_READ_WRITE);

  unsigned int histoSplit = 256;
  clContext.createBuffer("RadixSortSum", sizeof(unsigned int) * histoSplit, CL_MEM_READ_WRITE);
  clContext.createBuffer("RadixSortTempSum", sizeof(unsigned int) * histoSplit, CL_MEM_READ_WRITE);

  clContext.createBuffer("RadixSortIndicesIn", sizeof(unsigned int) * size, CL_MEM_READ_WRITE);
  clContext.fillBuffer("RadixSortIndicesIn", 0, sizeof(unsigned int) * m_indices.size(), m_indices.data());

  clContext.createBuffer("RadixSortIndicesOut", sizeof(unsigned int) * size, CL_MEM_READ_WRITE);
  return true;
}

bool RadixSort::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();
  /*
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_HISTOGRAM, { "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_SCAN, { "", "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_MERGE, { "", "", "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_REORDER, { "", "" });
*/
  return true;
}

void RadixSort::sort()
{
  CL::Context& clContext = CL::Context::Get();

  int nbRadixPasses = m_numTotalBits / m_numRadixBits;
  for (int radixPass = 0; radixPass < nbRadixPasses; ++radixPass)
  { /*
    clContext.runKernel(KERNEL_HISTOGRAM, m_numEntities);
    clContext.runKernel(KERNEL_SCAN, m_numEntities);
    clContext.runKernel(KERNEL_MERGE, m_numEntities);
    clContext.runKernel(KERNEL_REORDER, m_numEntities);*/
  }
}