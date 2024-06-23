
#pragma once

#include <array>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Utils
{
enum TaskState
{
  TS_BEGIN,
  TS_STOPPED,
  TS_RUNNING,
  TS_COMPLETED,
  TS_END,
  TS_AFTER_END,
  TS_INVALID = -1,
};

// map TaskState values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(TaskState, {
                                            { TS_INVALID, nullptr },
                                            { TS_BEGIN, "b" },
                                            { TS_STOPPED, "stopped" },
                                            { TS_RUNNING, "running" },
                                            { TS_COMPLETED, "completed" },
                                            { TS_END, "e" },
                                        })

// List of supported particles system sizes
enum NbParticles
{
  P512 = 1 << 9,
  P1K = 1 << 10,
  P4K = 1 << 12,
  P8K = 1 << 13,
  P16K = 1 << 14,
  P32K = 1 << 15,
  P65K = 1 << 16,
  P130K = 1 << 17
};

struct CompareNbParticles
{
  bool operator()(const NbParticles& nbA, const NbParticles& nbB) const
  {
    return (int)nbA < (int)nbB;
  }
};

struct NbParticlesInfo
{
  const std::string name;
  const std::array<int, 2> subdiv2D;
  const std::array<int, 3> subdiv3D;
};

static const std::map<NbParticles, NbParticlesInfo, CompareNbParticles> ALL_NB_PARTICLES {
  { NbParticles::P512, { "512", { 32, 16 }, { 8, 8, 8 } } },
  { NbParticles::P1K, { "1k", { 32, 32 }, { 16, 8, 8 } } },
  { NbParticles::P4K, { "4k", { 64, 64 }, { 16, 16, 16 } } },
  { NbParticles::P8K, { "8k", { 128, 64 }, { 32, 16, 16 } } },
  { NbParticles::P16K, { "16k", { 128, 128 }, { 32, 32, 16 } } },
  { NbParticles::P32K, { "32k", { 128, 256 }, { 32, 32, 32 } } },
  { NbParticles::P65K, { "65k", { 256, 256 }, { 64, 32, 32 } } },
  { NbParticles::P130K, { "130k", { 256, 512 }, { 64, 64, 32 } } }
};

static std::array<int, 2> GetNbParticlesSubdiv2D(NbParticles nbParts)
{
  const auto& it = ALL_NB_PARTICLES.find(nbParts);
  if (it != ALL_NB_PARTICLES.end())
    return it->second.subdiv2D;
  else
    return { 0, 0 };
};

static std::array<int, 3> GetNbParticlesSubdiv3D(NbParticles nbParts)
{
  const auto& it = ALL_NB_PARTICLES.find(nbParts);
  if (it != ALL_NB_PARTICLES.end())
    return it->second.subdiv3D;
  else
    return { 0, 0, 0 };
};

}