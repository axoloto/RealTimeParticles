#include "Engine.hpp"
#include "GLSL.hpp"
#include "Logging.hpp"
#include "Math.hpp"

using namespace Render;

Engine::Engine(EngineParams params)
    : m_maxNbParticles(params.maxNbParticles)
    , m_nbParticles(params.currNbParticles)
    , m_boxSize(params.boxSize)
    , m_gridRes(params.gridRes)
    , m_pointSize(params.pointSize)
    , m_isBoxVisible(true)
    , m_isGridVisible(false)
    , m_targetPos({ 0.0f, 0.0f, 0.0f })
    , m_dimension(params.dimension)
{
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_MULTISAMPLE);

  enableBlending(false);

  buildShaders();

  glGenVertexArrays(1, &m_VAO);
  glBindVertexArray(m_VAO);

  initCamera(params.aspectRatio);

  initPointCloud();

  initBox();

  initGrid();

  initTarget();
}

Engine::~Engine()
{
  glDeleteBuffers(1, &m_pointCloudCoordVBO);
  glDeleteBuffers(1, &m_pointCloudColorVBO);
  glDeleteBuffers(1, &m_box2DVBO);
  glDeleteBuffers(1, &m_box3DVBO);
  glDeleteBuffers(1, &m_cameraVBO);
  glDeleteBuffers(1, &m_targetVBO);
}

void Engine::buildShaders()
{
  m_pointCloudShader = std::make_unique<Shader>(Render::PointCloudVertShader, Render::PointCloudFragShader);
  m_box2DShader = std::make_unique<Shader>(Render::Box2DVertShader, Render::FragShader);
  m_box3DShader = std::make_unique<Shader>(Render::Box3DVertShader, Render::FragShader);
  m_gridShader = std::make_unique<Shader>(Render::GridVertShader, Render::FragShader);
  m_targetShader = std::make_unique<Shader>(Render::TargetVertShader, Render::FragShader);
}

void Engine::initCamera(float sceneAspectRatio)
{
  m_camera = std::make_unique<Camera>(sceneAspectRatio);

  // Filled at each frame, for OpenCL use
  glGenBuffers(1, &m_cameraVBO);
  loadCameraPos();
}

void Engine::initPointCloud()
{
  // Filled by OpenCL
  glGenBuffers(1, &m_pointCloudCoordVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudCoordVBO);
  glBufferData(GL_ARRAY_BUFFER, 4 * m_maxNbParticles * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(m_pointCloudPosAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Filled by OpenCL
  glGenBuffers(1, &m_pointCloudColorVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudColorVBO);
  glBufferData(GL_ARRAY_BUFFER, 4 * m_maxNbParticles * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(m_pointCloudColAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_pointCloudColAttribIndex);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Engine::draw()
{
  loadCameraPos();

  if (m_isBoxVisible)
    drawBox();

  if (m_isGridVisible)
    drawGrid();

  drawPointCloud();

  if (m_isTargetVisible)
    drawTarget();

  glFlush();
  glFinish();
}

void Engine::loadCameraPos()
{
  if (!m_camera)
    return;

  if (m_camera->isAutoRotating())
  {
    const auto angle = Math::float2(0.5f, 0.0f) * Math::PI_F / 180.0f * 0.5f;
    m_camera->rotate(angle.y, angle.x);
  }

  auto pos = m_camera->cameraPos();
  const std::array<float, 3> cameraCoord = { pos[0], pos[1], pos[2] };
  glBindBuffer(GL_ARRAY_BUFFER, m_cameraVBO);
  glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), &cameraCoord, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Engine::drawPointCloud()
{
  m_pointCloudShader->activate();

  m_pointCloudShader->setUniform("u_pointSize", (int)m_pointSize);
  m_pointCloudShader->setUniform("u_projView", m_camera->getProjViewMat());
  m_pointCloudShader->setUniform("u_cameraPos", m_camera->cameraPos());

  glDrawArrays(GL_POINTS, 0, (GLsizei)m_nbParticles);

  m_pointCloudShader->deactivate();
}

void Engine::drawBox()
{
  if (m_dimension == Geometry::Dimension::dim2D)
  {
    m_box2DShader->activate();

    m_box2DShader->setUniform("u_projView", m_camera->getProjViewMat());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_box2DEBO);
    glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_box2DShader->deactivate();
  }
  else
  {
    m_box3DShader->activate();

    m_box3DShader->setUniform("u_projView", m_camera->getProjViewMat());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_box3DEBO);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_box3DShader->deactivate();
  }
}

void Engine::drawGrid()
{
  m_gridShader->activate();

  m_gridShader->setUniform("u_projView", m_camera->getProjViewMat());

  GLsizei numGridCells = (GLsizei)(m_gridRes.x * m_gridRes.y * m_gridRes.z);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gridEBO);
  glDrawElements(GL_LINES, 24 * numGridCells, GL_UNSIGNED_INT, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  m_gridShader->deactivate();
}

void Engine::drawTarget()
{
  m_targetShader->activate();

  m_targetShader->setUniform("u_projView", m_camera->getProjViewMat());

  const std::array<float, 3> targetCoord = { m_targetPos[0], m_targetPos[1], m_targetPos[2] };
  glBindBuffer(GL_ARRAY_BUFFER, m_targetVBO);
  glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), &targetCoord, GL_DYNAMIC_DRAW);
  glDrawArrays(GL_POINTS, 0, (GLsizei)1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  m_targetShader->deactivate();
}

void Engine::initBox()
{
  // 2D
  auto box2DVertices = Geometry::RefSquareVertices;
  for (auto& vertex : box2DVertices)
  {
    float y = vertex[0] * m_boxSize.y / 2.0f;
    float z = vertex[1] * m_boxSize.z / 2.0f;
    vertex = { y, z };
  }

  glGenBuffers(1, &m_box2DVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_box2DVBO);
  glVertexAttribPointer(m_box2DPosAttribIndex, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_box2DPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, sizeof(box2DVertices.front()) * box2DVertices.size(), box2DVertices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &m_box2DEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_box2DEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Geometry::RefSquareIndices), Geometry::RefSquareIndices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // 3D
  auto box3DVertices = Geometry::RefCubeVertices;
  for (auto& vertex : box3DVertices)
  {
    float x = vertex[0] * m_boxSize.x / 2.0f;
    float y = vertex[1] * m_boxSize.y / 2.0f;
    float z = vertex[2] * m_boxSize.z / 2.0f;
    vertex = { x, y, z };
  }

  glGenBuffers(1, &m_box3DVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_box3DVBO);
  glVertexAttribPointer(m_box3DPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_box3DPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, sizeof(box3DVertices.front()) * box3DVertices.size(), box3DVertices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &m_box3DEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_box3DEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Geometry::RefCubeIndices), Geometry::RefCubeIndices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Engine::initGrid()
{
  Geometry::Vertex3D cellDims;
  cellDims[0] = (float)m_boxSize.x / m_gridRes.x;
  cellDims[1] = (float)m_boxSize.y / m_gridRes.y;
  cellDims[2] = (float)m_boxSize.z / m_gridRes.z;

  auto localCellCoords = Geometry::RefCubeVertices;
  for (auto& vertex : localCellCoords)
  {
    float x = vertex[0] * cellDims[0] * 0.5f;
    float y = vertex[1] * cellDims[1] * 0.5f;
    float z = vertex[2] * cellDims[2] * 0.5f;
    vertex = { x, y, z };
  }

  size_t centerIndex = 0;
  size_t numCells = m_gridRes.x * m_gridRes.y * m_gridRes.z;
  Geometry::Vertex3D firstPos;
  firstPos[0] = -(float)m_boxSize.x / 2.0f + 0.5f * cellDims[0];
  firstPos[1] = -(float)m_boxSize.y / 2.0f + 0.5f * cellDims[1];
  firstPos[2] = -(float)m_boxSize.z / 2.0f + 0.5f * cellDims[2];
  std::vector<Geometry::Vertex3D> globalCellCenterCoords(numCells);
  for (int x = 0; x < m_gridRes.x; ++x)
  {
    float xCoord = firstPos[0] + x * cellDims[0];
    for (int y = 0; y < m_gridRes.y; ++y)
    {
      float yCoord = firstPos[1] + y * cellDims[1];
      for (int z = 0; z < m_gridRes.z; ++z)
      {
        float zCoord = firstPos[2] + z * cellDims[2];
        globalCellCenterCoords.at(centerIndex++) = { xCoord, yCoord, zCoord };
      }
    }
  }

  size_t cornerIndex = 0;
  std::vector<Geometry::Vertex3D> globalCellCoords(numCells * 8);
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

  glGenBuffers(1, &m_gridPosVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_gridPosVBO);
  glVertexAttribPointer(m_gridPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_gridPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, sizeof(globalCellCoords.front()) * globalCellCoords.size(), globalCellCoords.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Filled by OpenCL
  glGenBuffers(1, &m_gridDetectorVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_gridDetectorVBO);
  glVertexAttribPointer(m_gridDetectorAttribIndex, 1, GL_FLOAT, GL_FALSE, sizeof(float), nullptr);
  glEnableVertexAttribArray(m_gridDetectorAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, 8 * numCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  size_t index = 0;
  GLuint globalOffset = 0;
  std::vector<GLuint> globalCellIndices(numCells * 24);
  for (const auto& centerCoords : globalCellCenterCoords)
  {
    for (const auto& localIndex : Geometry::RefCubeIndices)
    {
      globalCellIndices.at(index++) = (GLuint)localIndex + globalOffset;
    }
    globalOffset += 8;
  }

  size_t gridIndexSize = sizeof(globalCellIndices.front()) * globalCellIndices.size();
  glGenBuffers(1, &m_gridEBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gridEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndexSize, globalCellIndices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Engine::initTarget()
{
  const std::array<float, 3> targetCoord = { m_targetPos[0], m_targetPos[1], m_targetPos[2] };
  glGenBuffers(1, &m_targetVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_targetVBO);
  glVertexAttribPointer(m_targetPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
  glEnableVertexAttribArray(m_targetPosAttribIndex);
  glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), &targetCoord, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Engine::checkMouseEvents(UserAction action, Math::float2 delta)
{
  switch (action)
  {
  case UserAction::TRANSLATION:
  {
    const auto displacement = 0.06f * delta;
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
    m_camera->zoom(0.5f * delta.x);
    break;
  }
  }
}

void Engine::enableBlending(bool enable)
{
  m_isBlendingEnabled = enable;

  if (m_isBlendingEnabled)
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
  }
  else
  {
    glDisable(GL_BLEND);
  }
}
