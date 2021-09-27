
#include "Geometry.hpp"
#include "Logging.hpp"

namespace Geometry
{
std::vector<Math::float3> Generate2DGrid(Shape shape, Plane plane,
    Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (gridRes.x * gridRes.y <= 0)
  {
    LOG_ERROR("Cannot generate grid with negative or null number of vertices");
    std::vector<Math::float3> nullVec;
    return nullVec;
  }

  std::vector<Math::float3> verts(gridRes.x * gridRes.y, Math::float3(0.0f, 0.0f, 0.0f));
  switch (shape)
  {
  case Shape::Circle:
    GenerateCircularGrid(plane, verts, gridRes, gridStartPos, gridEndPos);
    break;
  case Shape::Rectangle:
    GenerateRectangularGrid(plane, verts, gridRes, gridStartPos, gridEndPos);
    break;
  default:
    break;
  }

  return verts;
}

std::vector<Math::float3> Generate3DGrid(Shape shape,
    Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (gridRes.x * gridRes.y * gridRes.z <= 0)
  {
    LOG_ERROR("Cannot generate grid with negative or null number of vertices");
    std::vector<Math::float3> nullVec;
    return nullVec;
  }

  std::vector<Math::float3> verts(gridRes.x * gridRes.y * gridRes.z, Math::float3(0.0f, 0.0f, 0.0f));
  switch (shape)
  {
  case Shape::Sphere:
    GenerateSphereGrid(verts, gridRes, gridStartPos, gridEndPos);
    break;
  case Shape::Box:
    GenerateBoxGrid(verts, gridRes, gridStartPos, gridEndPos);
  default:
    break;
  }

  return verts;
}

void GenerateRectangularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  Math::float3 vec = Math::abs(gridStartPos - gridEndPos);

  Math::float3 gridSpacing;
  Math::int3 gridResExt;

  switch (plane)
  {
  case Plane::XY:
    gridSpacing = Math::float3({ vec.x / gridRes.x, vec.y / gridRes.y, 0.0f });
    gridResExt = Math::int3({ gridRes.x, gridRes.y, 1 });
    break;
  case Plane::XZ:
    gridSpacing = Math::float3({ vec.x / gridRes.x, 0.0f, vec.z / gridRes.y });
    gridResExt = Math::int3({ gridRes.x, 1, gridRes.y });
    break;
  case Plane::YZ:
    gridSpacing = Math::float3({ 0.0f, vec.y / gridRes.x, vec.z / gridRes.y });
    gridResExt = Math::int3({ 1, gridRes.x, gridRes.y });
    break;
  default:
    LOG_ERROR("Cannot generate rectangular grid. Plane not existing");
    break;
  }

  int vertIndex = 0;
  for (int ix = 0; ix < gridResExt.x; ++ix)
  {
    for (int iy = 0; iy < gridResExt.y; ++iy)
    {
      for (int iz = 0; iz < gridResExt.z; ++iz)
      {
        verts[vertIndex++] = { gridStartPos.x + ix * gridSpacing.x,
          gridStartPos.y + iy * gridSpacing.y,
          gridStartPos.z + iz * gridSpacing.z };
      }
    }
  }
}

void GenerateCircularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
}

void GenerateBoxGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
}

void GenerateSphereGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
}

}