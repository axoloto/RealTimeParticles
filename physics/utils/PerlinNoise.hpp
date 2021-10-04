#pragma once

#include <array>
#include <vector>

#include "Math.hpp"
namespace Physics
{
// Directly inspired by Ken Perlin's work and https://solarianprogrammer.com/ cpp implementation

class PerlinNoise
{
  public:
  PerlinNoise(int seed);
  ~PerlinNoise() = default;

  float computeNoiseValue(Math::float3 pos);

  private:
  // The permutation vector
  std::vector<int> m_perms;

  float fade(float t);
  float lerp(float t, float a, float b);
  float grad(int hash, float x, float y, float z);
};
}
