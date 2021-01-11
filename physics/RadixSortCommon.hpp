#ifndef OCLRADIXSORT_TESTCOMMON_HPP_H
#define OCLRADIXSORT_TESTCOMMON_HPP_H

#include <algorithm>
#include <chrono>
#include <random>

#include "tpRadixSort.hpp"

// change this to select desired platform and device for all tests
#define PLATFORM_DEFAULT_ID 0
#define DEVICE_DEFAULT_ID 0

// helper function for obtaining CL device
cl::Device getDevice(int pid = PLATFORM_DEFAULT_ID, int did = DEVICE_DEFAULT_ID, cl_uint type = CL_DEVICE_TYPE_GPU)
{
  std::vector<cl::Platform> pls;
  cl::Platform::get(&pls);

  std::vector<cl::Device> dvs;
  pls[pid].getDevices(type, &dvs);

  for (auto& d : dvs)
  {
    std::cout << d.getInfo<CL_DEVICE_NAME>() << ' ' << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
  }

  return dvs[did];
}

// helper function to check the correctness of permutation
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

// sample Radix Sort configuration
// type of keys
using T = unsigned /*long long*/ int;
// type of indices
using TI = T;
// radix sort template configuration
using SimpleRadixSort64bit = RadixSort<8, 32, T, TI, 128, 4, true, 256, false, 32, true>;

#endif //OCLRADIXSORT_TESTCOMMON_HPP_H