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

  size_t m_numEntities;
};
}