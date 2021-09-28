
#include "Geometry.hpp"
#include "Logging.hpp"
#include <cmath>

namespace Geometry
{
std::vector<Math::float3> Generate2DGrid(Shape2D shape, Plane plane,
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
  case Shape2D::Circle:
    GenerateCircularGrid(plane, verts, gridRes, gridStartPos, gridEndPos);
    break;
  case Shape2D::Rectangle:
    GenerateRectangularGrid(plane, verts, gridRes, gridStartPos, gridEndPos);
    break;
  default:
    break;
  }

  return verts;
}

std::vector<Math::float3> Generate3DGrid(Shape3D shape,
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
  case Shape3D::Sphere:
    GenerateSphereGrid(verts, gridRes, gridStartPos, gridEndPos);
    break;
  case Shape3D::Box:
    GenerateBoxGrid(verts, gridRes, gridStartPos, gridEndPos);
  default:
    break;
  }

  return verts;
}

void GenerateRectangularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (verts.size() < gridRes.x * gridRes.y)
  {
    LOG_ERROR("Cannot generate grid with this resolution");
    return;
  }

  Math::float3 vec = gridEndPos - gridStartPos;

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
        verts[vertIndex++] = {
          gridStartPos.x + ix * gridSpacing.x,
          gridStartPos.y + iy * gridSpacing.y,
          gridStartPos.z + iz * gridSpacing.z
        };
      }
    }
  }
}

void GenerateCircularGrid(Plane plane, std::vector<Math::float3>& verts, Math::int2 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (verts.size() < gridRes.x * gridRes.y)
  {
    LOG_ERROR("Cannot generate grid with this resolution");
    return;
  }

  Math::float3 vec = gridEndPos - gridStartPos;
  Math::float3 gridCenterPos = gridStartPos + vec / 2.0f;
  float radius = Math::length(vec) / 2.0f;
  float angleSpacing = 2.0f * Math::PI_F / gridRes.x;
  float radiusSpacing = radius / gridRes.y;
  int vertIndex = 0;

  switch (plane)
  {
  case Plane::XY:
    for (int io = 0; io < gridRes.x; ++io)
    {
      for (int ir = 1; ir < gridRes.y + 1; ++ir)
      {
        verts[vertIndex++] = {
          gridCenterPos.x + (ir * radiusSpacing) * std::cos(io * angleSpacing),
          gridCenterPos.y + (ir * radiusSpacing) * std::sin(io * angleSpacing),
          gridCenterPos.z
        };
      }
    }
    break;
  case Plane::XZ:
    for (int io = 0; io < gridRes.x; ++io)
    {
      for (int ir = 1; ir < gridRes.y + 1; ++ir)
      {
        verts[vertIndex++] = {
          gridCenterPos.x + (ir * radiusSpacing) * std::cos(io * angleSpacing),
          gridCenterPos.y,
          gridCenterPos.z + (ir * radiusSpacing) * std::sin(io * angleSpacing)
        };
      }
    }
    break;
  case Plane::YZ:
    for (int io = 0; io < gridRes.x; ++io)
    {
      for (int ir = 0; ir < gridRes.y; ++ir)
      {
        verts[vertIndex++] = {
          gridCenterPos.x,
          gridCenterPos.y + ((ir + 1) * radiusSpacing) * std::cos(io * angleSpacing),
          gridCenterPos.z + ((ir + 1) * radiusSpacing) * std::sin(io * angleSpacing)
        };
      }
    }
    break;
  default:
    LOG_ERROR("Cannot generate rectangular grid. Plane not existing");
    break;
  }
}

void GenerateBoxGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (verts.size() < gridRes.x * gridRes.y * gridRes.z)
  {
    LOG_ERROR("Cannot generate grid with this resolution");
    return;
  }

  Math::float3 vec = gridEndPos - gridStartPos;
  Math::float3 gridSpacing = Math::float3({ vec.x / gridRes.x, vec.y / gridRes.y, vec.z / gridRes.z });

  int vertIndex = 0;
  for (int ix = 0; ix < gridRes.x; ++ix)
  {
    for (int iy = 0; iy < gridRes.y; ++iy)
    {
      for (int iz = 0; iz < gridRes.z; ++iz)
      {
        verts[vertIndex++] = {
          gridStartPos.x + ix * gridSpacing.x,
          gridStartPos.y + iy * gridSpacing.y,
          gridStartPos.z + iz * gridSpacing.z
        };
      }
    }
  }
}

void GenerateSphereGrid(std::vector<Math::float3>& verts, Math::int3 gridRes, Math::float3 gridStartPos, Math::float3 gridEndPos)
{
  if (verts.size() < gridRes.x * gridRes.y * gridRes.z)
  {
    LOG_ERROR("Cannot generate grid with this resolution");
    return;
  }

  Math::float3 vec = gridEndPos - gridStartPos;
  Math::float3 gridCenterPos = gridStartPos + vec / 2.0f;
  float radius = Math::length(vec) / 2.0f;
  float phiSpacing = Math::PI_F / gridRes.x;
  float thetaSpacing = 2.0f * Math::PI_F / gridRes.y;
  float radiusSpacing = radius / gridRes.z;

  int vertIndex = 0;
  for (int iphi = 0; iphi < gridRes.x; ++iphi)
  {
    for (int itheta = 0; itheta < gridRes.y; ++itheta)
    {
      for (int ir = 0; ir < gridRes.z; ++ir)
      {
        verts[vertIndex++] = {
          gridCenterPos.x + ((ir + 1) * radiusSpacing) * std::cos(itheta * thetaSpacing) * std::sin(iphi * phiSpacing),
          gridCenterPos.y + ((ir + 1) * radiusSpacing) * std::sin(itheta * thetaSpacing) * std::sin(iphi * phiSpacing),
          gridCenterPos.z + ((ir + 1) * radiusSpacing) * std::cos(iphi * phiSpacing),
        };
      }
    }
  }
}
}