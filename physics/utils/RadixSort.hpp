#pragma once

#include <array>
#include <vector>

#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>

namespace Physics
{
template <typename T, typename U>
bool checkPermutation(const std::vector<T>& keysAfterSort,
    const std::vector<T>& keysBeforeSort,
    const std::vector<U>& permutation)
{
  for (size_t i = 0; i < keysAfterSort.size(); ++i)
  {
    if (keysBeforeSort[permutation[i]] != keysAfterSort[i])
    {
      std::cout << i << " " << keysAfterSort[i] << " " << keysBeforeSort[i] << " " << permutation[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
auto makeRng(T upperBound)
{
  return std::linear_congruential_engine<T, std::minstd_rand::multiplier, std::minstd_rand::increment, std::minstd_rand::modulus> {
    static_cast<T>(std::chrono::steady_clock::now().time_since_epoch().count())
  };
}
class RadixSort
{
  public:
  RadixSort(size_t numEntities);
  ~RadixSort() = default;

  void sort(const std::string& inputKeyBufferName, const std::vector<std::string>& optionalInputBufferNames = {});

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  size_t m_numEntities;

  unsigned int m_numRadix;

  unsigned int m_numRadixBits;
  unsigned int m_numTotalBits;

  unsigned int m_numGroups;
  unsigned int m_numItems;

  size_t m_histoSplit;

  int m_numRadixPasses;

  std::vector<unsigned int> m_indices;
};
}