#pragma once

#include "Math.hpp"
#include "Parameters.hpp"

#include <cstdint>
#include <vector>

namespace Geometry
{

enum class Dimension
{
  dim2D,
  dim3D
};

enum class Shape2D
{
  Rectangle,
  Circle
};

enum class Shape3D
{
  Box,
  Sphere
};

enum class Plane
{
  XY,
  YZ,
  XZ
};

// 3D Box
typedef std::array<float, 3> Vertex3D;
constexpr std::array<Vertex3D, 8> RefCubeVertices {
  Vertex3D({ 1.f, -1.f, -1.f }),
  Vertex3D({ 1.f, 1.f, -1.f }),
  Vertex3D({ -1.f, 1.f, -1.f }),
  Vertex3D({ -1.f, -1.f, -1.f }),
  Vertex3D({ 1.f, -1.f, 1.f }),
  Vertex3D({ 1.f, 1.f, 1.f }),
  Vertex3D({ -1.f, 1.f, 1.f }),
  Vertex3D({ -1.f, -1.f, 1.f })
};

constexpr std::array<std::uint32_t, 24> RefCubeIndices {
  0, 1,
  1, 2,
  2, 3,
  3, 0,
  4, 5,
  5, 6,
  6, 7,
  7, 4,
  0, 4,
  1, 5,
  2, 6,
  3, 7
};

// 2D Box
typedef std::array<float, 2> Vertex2D;
constexpr std::array<Vertex2D, 4> RefSquareVertices {
  Vertex2D({ -1.f, -1.f }),
  Vertex2D({ -1.f, 1.f }),
  Vertex2D({ 1.f, 1.f }),
  Vertex2D({ 1.f, -1.f }),
};

constexpr std::array<std::uint32_t, 24> RefSquareIndices {
  0, 1,
  1, 2,
  2, 3,
  3, 0
};

std::vector<Math::float3> Generate2DGrid(Shape2D shape, Plane plane,
    Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);

std::vector<Math::float3> Generate3DGrid(Shape3D shape,
    Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);

void GenerateRectangularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateCircularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateBoxGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
void GenerateSphereGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos);
}