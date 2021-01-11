#include "RadixSort.hpp"

#include <ctime>
#include <iostream>
#include <spdlog/spdlog.h>

using namespace Core;

#define PROGRAM_RADIXSORT "RadixSort"

#define KERNEL_HISTOGRAM "histogram"
#define KERNEL_MERGE "merge"
#define KERNEL_SCAN "scan"
#define KERNEL_REORDER "reorder"

RadixSort::RadixSort(size_t numEntities)
    : m_numEntities(numEntities)
{
  createProgram();

  createBuffers();

  createKernels();
}

bool RadixSort::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  std::string buildOptions = "-D_RADIX=128 -D_BITS=8 -D_GROUPS=128 -D_ITEMS=4 -D_TILESIZE=32 -DCOMPUTE_PERMUTATION";
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

  //size_t bufferSize = 4 * NUM_MAX_ENTITIES * sizeof();

  // clContext.createBuffer("RadixSortVel", RadixSortBufferSize, CL_MEM_READ_WRITE);
  // clContext.createBuffer("RadixSortAcc", RadixSortBufferSize, CL_MEM_READ_WRITE);
  // clContext.createBuffer("RadixSortParams", sizeof(m_RadixSortParams), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
  // clContext.createBuffer("gridParams", sizeof(m_gridParams), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);

  return true;
}

bool RadixSort::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_HISTOGRAM, { "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_SCAN, { "", "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_MERGE, { "", "", "" });
  clContext.createKernel(PROGRAM_RADIXSORT, KERNEL_REORDER, { "", "" });

  return true;
}

void RadixSort::sort()
{
  CL::Context& clContext = CL::Context::Get();
  /*
  clContext.runKernel(KERNEL_HISTOGRAM, m_numEntities);
  clContext.runKernel(KERNEL_SCAN, m_numEntities);
  clContext.runKernel(KERNEL_MERGE, m_numEntities);
  clContext.runKernel(KERNEL_REORDER, m_numEntities);
  */
}