#pragma once

//#include "CL/cl.h"
#include <array>
#include <vector>

#include "Physics.hpp"
#include "ocl/Context.hpp"

namespace Core
{
class RadixSort
{
  public:
  RadixSort(size_t numEntities);
  ~RadixSort() = default;

  void sort();

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  unsigned int m_numEntities;

  unsigned int m_numRadix;

  unsigned int m_numRadixBits;
  unsigned int m_numTotalBits;

  unsigned int m_numGroups;
  unsigned int m_numItems;

  unsigned int m_numRadixPasses;

  std::vector<unsigned int> m_indices;
};
}