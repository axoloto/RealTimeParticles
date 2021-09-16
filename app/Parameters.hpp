
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace App
{
// List of supported particles system sizes
enum NbParticles
{
  P512 = 1 << 9,
  P1K = 1 << 10,
  P4K = 1 << 12,
  P65K = 1 << 16,
  P130K = 1 << 17,
  P260K = 1 << 18
};

struct CompareNbParticles
{
  bool operator()(const NbParticles& nbA, const NbParticles& nbB) const
  {
    return (int)nbA < (int)nbB;
  }
};

static const std::map<NbParticles, std::string, CompareNbParticles> ALL_NB_PARTICLES {
  { NbParticles::P512, "512" },
  { NbParticles::P1K, "1k" },
  { NbParticles::P4K, "4k" },
  { NbParticles::P65K, "65k" },
  { NbParticles::P130K, "130k" },
  { NbParticles::P260K, "260k" }
};

// Length of one side of the bounding box where the particles evolve
static constexpr int BOX_SIZE = 200;

// Length of one side of the cells forming the 3D grid containing all the particles
static constexpr int GRID_RES = 50;

}