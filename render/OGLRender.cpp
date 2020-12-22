#include "GLSL.hpp"
#include "Math.hpp"
#include "OGLRender.hpp"

using namespace Render;

OGLRender::OGLRender(size_t boxSize, size_t numDisplayedEntities, size_t numMaxEntities, float sceneAspectRatio)
    : m_boxSize(boxSize)
    , m_boxNumDivs(10)
    , m_numDisplayedEntities(numDisplayedEntities)
    , m_numMaxEntities(numMaxEntities)
    , m_isBoxVisible(true)
    , m_isGridVisible(true)
{
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  initCamera(sceneAspectRatio);

  buildShaders();

  connectPointCloudVBOsToVAO();

  generateBox();

  generateGrid();
}

OGLRender::~OGLRender()
{
  glDeleteBuffers(1, &m_pointCloudCoordVBO);
  glDeleteBuffers(1, &m_pointCloudColorVBO);
  glDeleteBuffers(1, &m_boxVBO);
}

void OGLRender::initCamera(float sceneAspectRatio)
{
  m_camera = std::make_unique<Camera>(sceneAspectRatio);
}

void OGLRender::buildShaders()
{
  m_pointCloudShader = std::make_unique<OGLShader>(Render::PointCloudVertShader, Render::FragShader);
  m_boxShader = std::make_unique<OGLShader>(Render::BoxVertShader, Render::FragShader);
  m_gridShader = std::make_unique<OGLShader>(Render::GridVertShader, Render::FragShader);
}

void OGLRender::connectPointCloudVBOsToVAO()
{
  glGenVertexArrays(1, &m_VAO);
  glBindVertexArray(m_VAO);

  glGenBuffers(1, &m_pointCloudCoordVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudCoordVBO);
  glBufferData(GL_ARRAY_BUFFER, 4 * m_numMaxEntities * sizeof(float), NULL, GL_DYNAMIC_DRAW); // WIP
  glVertexAttribPointer(m_pointCloudPosAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
  glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &m_pointCloudColorVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudColorVBO);
  glBufferData(GL_ARRAY_BUFFER, 4 * m_numMaxEntities * sizeof(float), NULL, GL_DYNAMIC_DRAW); // WIP
  glVertexAttribPointer(m_pointCloudColAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
  glEnableVertexAttribArray(m_pointCloudColAttribIndex);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OGLRender::draw()
{
  drawPointCloud();

  if (m_isBoxVisible)
    drawBox();

  if (m_isGridVisible)
    drawGrid();
}

void OGLRender::drawPointCloud()
{
  m_pointCloudShader->activate();

  Math::float4x4 projViewMat = m_camera->getProjViewMat();
  m_pointCloudShader->setUniform("u_projView", projViewMat);

  glDrawArrays(GL_POINTS, 0, (GLsizei)m_numDisplayedEntities);

  m_pointCloudShader->deactivate();
}

void OGLRender::drawBox()
{
  m_boxShader->activate();

  Math::float4x4 projViewMat = m_camera->getProjViewMat();
  m_boxShader->setUniform("u_projView", projViewMat);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEBO);
  glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  m_boxShader->deactivate();
}

void OGLRender::drawGrid()
{
  m_gridShader->activate();

  Math::float4x4 projViewMat = m_camera->getProjViewMat();
  m_gridShader->setUniform("u_projView", projViewMat);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gridEBO);
  glDrawElements(GL_LINES, 24000, GL_UNSIGNED_INT, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  m_gridShader->deactivate();
}

void OGLRender::generateBox()
{
  std::array<Vertex, 8> boxVertices = m_refCubeVertices;
  for (auto& vertex : boxVertices)
  {
    float x = vertex[0] * m_boxSize / 2.0f;
    float y = vertex[1] * m_boxSize / 2.0f;
    float z = vertex[2] * m_boxSize / 2.0f;
    vertex = { x, y, z };
  }

  size_t boxBufferSize = sizeof(boxVertices[0]) * boxVertices.size();
  glGenBuffers(1, &m_boxVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
  glVertexAttribPointer(m_boxPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
  glEnableVertexAttribArray(m_boxPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, boxBufferSize, boxVertices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &m_boxEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_refCubeIndices), m_refCubeIndices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OGLRender::generateGrid()
{
  float res = 1.0f * m_boxSize / m_boxNumDivs;
  std::array<Vertex, 8> localCellCoords = m_refCubeVertices;
  for (auto& vertex : localCellCoords)
  {
    float x = vertex[0] * res * 0.5f;
    float y = vertex[1] * res * 0.5f;
    float z = vertex[2] * res * 0.5f;
    vertex = { x, y, z };
  }

  size_t centerIndex = 0;
  std::array<Vertex, 1000> globalCellCenterCoords;
  for (int x = 0; x < 10; ++x)
  {
    float xCoord = -(float)m_boxSize / 2.0f + (x + 0.5f) * res;
    for (int y = 0; y < 10; ++y)
    {
      float yCoord = -(float)m_boxSize / 2.0f + (y + 0.5f) * res;
      for (int z = 0; z < 10; ++z)
      {
        float zCoord = -(float)m_boxSize / 2.0f + (z + 0.5f) * res;
        globalCellCenterCoords.at(centerIndex++) = { xCoord, yCoord, zCoord };
      }
    }
  }

  size_t cornerIndex = 0;
  std::array<Vertex, 8000> globalCellCoords;
  for (const auto& centerCoords : globalCellCenterCoords)
  {
    for (const auto& cornerCoords : localCellCoords)
    {
      globalCellCoords.at(cornerIndex)[0] = cornerCoords[0] + centerCoords[0];
      globalCellCoords.at(cornerIndex)[1] = cornerCoords[1] + centerCoords[1];
      globalCellCoords.at(cornerIndex)[2] = cornerCoords[2] + centerCoords[2];
      ++cornerIndex;
    }
  }

  size_t gridBufferSize = sizeof(globalCellCoords[0]) * globalCellCoords.size();
  glGenBuffers(1, &m_gridVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);
  glVertexAttribPointer(m_gridPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
  glEnableVertexAttribArray(m_gridPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, gridBufferSize, &globalCellCoords[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  size_t index = 0;
  GLuint globalOffset = 0;
  std::array<GLuint, 24000> globalCellIndices;
  for (const auto& centerCoords : globalCellCenterCoords)
  {
    for (const auto& localIndex : m_refCubeIndices)
    {
      globalCellIndices.at(index++) = localIndex + globalOffset;
    }
    globalOffset += 8;
  }

  glGenBuffers(1, &m_gridEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gridEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(globalCellIndices), &globalCellIndices[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OGLRender::checkMouseEvents(UserAction action, Math::float2 delta)
{
  switch (action)
  {
  case UserAction::TRANSLATION:
  {
    const auto displacement = 0.4f * delta;
    m_camera->translate(-displacement.x, displacement.y);
    break;
  }
  case UserAction::ROTATION:
  {
    const auto angle = delta * Math::PI_F / 180.0f * 0.5;
    m_camera->rotate(angle.y, angle.x);
    break;
  }
  case UserAction::ZOOM:
  {
    m_camera->zoom(delta.x);
    break;
  }
  }
}