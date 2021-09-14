
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace App
{
// List of supported physical models
enum PhysicsModel
{
  BOIDS = 0,
  FLUIDS = 1
};

struct ComparePhysicsModel
{
  bool operator()(const PhysicsModel& modelA, const PhysicsModel& modelB) const
  {
    return (int)modelA < (int)modelB;
  }
};

static const std::map<PhysicsModel, std::string, ComparePhysicsModel> ALL_PHYSICS_MODELS {
  { PhysicsModel::BOIDS, "Boids" },
  { PhysicsModel::FLUIDS, "Fluids" },
};

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

static const std::vector<std::pair<int, std::string>> ALL_POSSIBLE_NB_PARTS {
  std::make_pair(NbParticles::P512, "512"),
  std::make_pair(NbParticles::P1K, "1k"),
  std::make_pair(NbParticles::P4K, "4k"),
  std::make_pair(NbParticles::P65K, "65k"),
  std::make_pair(NbParticles::P130K, "130k"),
  std::make_pair(NbParticles::P260K, "260k")
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
static constexpr int BOX_SIZE = 1600;

// Length of one side of the cells forming the 3D grid containing all the particles
static constexpr int GRID_RES = 30;

// Helpers UI
std::string AllPossibleNbParts()
{
  std::string allNbParts;
  for (const auto& nbPart : ALL_POSSIBLE_NB_PARTS)
    allNbParts += nbPart.second + '\0';
  return allNbParts;
}

int FindNbPartsByIndex(size_t index)
{
  return (index < ALL_POSSIBLE_NB_PARTS.size()) ? ALL_POSSIBLE_NB_PARTS[index].first : 0;
}

int FindNbPartsIndex(int nbPart)
{
  for (int i = 0; i < ALL_POSSIBLE_NB_PARTS.size(); ++i)
  {
    if (ALL_POSSIBLE_NB_PARTS[i].first == nbPart)
      return i;
  }
  return 0;
}
}