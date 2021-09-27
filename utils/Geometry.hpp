#pragma once

#include "Math.hpp"
#include "Parameters.hpp"

#include <vector>

namespace Geometry
{
enum class Shape
{
  Rectangle,
  Circle,
  Box,
  Sphere
};

enum class Plane
{
  XY,
  YZ,
  XZ
};

std::vector<Math::float3> Generate2DGrid(Shape shape, Plane plane,
    Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);

std::vector<Math::float3> Generate3DGrid(Shape shape,
    Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);

void GenerateRectangularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateCircularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateBoxGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateSphereGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
}