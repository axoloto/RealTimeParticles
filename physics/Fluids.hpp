#pragma once

#include <array>
#include <chrono>
#include <vector>

#include "Physics.hpp"
#include "ocl/Context.hpp"
#include "utils/RadixSort.hpp"
#include "utils/Target.hpp"

namespace Core
{
using clock = std::chrono::high_resolution_clock;
class Fluids : public Model
{
  public:
  Fluids(ModelParams params);
  ~Fluids() = default;

  void update() override;
  void reset() override;

  //
  void setVelocity(float velocity) override
  {
    m_velocity = velocity;
    //updateFluidsParamsInKernel();
  }

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void updateFluidsParamsInKernel();

  bool m_activeAlignment;
  bool m_activeCohesion;
  bool m_activeSeparation;

  float m_scaleAlignment;
  float m_scaleCohesion;
  float m_scaleSeparation;

  bool m_simplifiedMode;
  size_t m_maxNbPartsInCell;

  std::unique_ptr<Target> m_target;

  RadixSort m_radixSort;
  std::chrono::steady_clock::time_point m_time;
};
}