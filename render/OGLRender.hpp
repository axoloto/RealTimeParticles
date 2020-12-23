#pragma once

#include "Camera.hpp"
#include "OGLShader.hpp"
#include <array>
#include <glad/glad.h>
#include <memory>
#include <vector>

namespace Render
{
enum class UserAction
{
  TRANSLATION,
  ROTATION,
  ZOOM
};

class OGLRender
{
  public:
  OGLRender(size_t boxSize, size_t gridRes, size_t numDisplayedEntities, size_t numMaxEntities, float aspectRatio);
  ~OGLRender();

  void checkMouseEvents(UserAction action, Math::float2 mouseDisplacement);
  void draw();

  inline const Math::float3 cameraPos() const { return m_camera->cameraPos(); }
  inline const Math::float3 targetPos() const { return m_camera->targetPos(); }

  inline void resetCamera() { m_camera->reset(); }
  inline void setWindowSize(Math::int2 windowSize)
  {
    if (m_camera)
      m_camera->setSceneAspectRatio((float)windowSize.x / windowSize.y);
  }
  inline void setNumDisplayedEntities(int numDisplayedEntities) { m_numDisplayedEntities = numDisplayedEntities; }

  bool isBoxVisible() const { return m_isBoxVisible; }
  void setBoxVisibility(bool isVisible) { m_isBoxVisible = isVisible; }

  bool isGridVisible() const { return m_isGridVisible; }
  void setGridVisibility(bool isVisible) { m_isGridVisible = isVisible; }

  GLuint pointCloudCoordVBO() const { return m_pointCloudCoordVBO; }
  GLuint pointCloudColorVBO() const { return m_pointCloudColorVBO; }
  GLuint gridColorVBO() const { return m_gridColVBO; }

  private:
  void buildShaders();
  void connectPointCloudVBOsToVAO();

  void drawPointCloud();

  void generateBox();
  void drawBox();

  void generateGrid();
  void drawGrid();

  void initCamera(float sceneAspectRatio);

  const GLuint
      m_pointCloudPosAttribIndex { 0 },
      m_pointCloudColAttribIndex { 1 },
      m_boxPosAttribIndex { 2 },
      m_gridPosAttribIndex { 3 },
      m_gridColAttribIndex { 4 };

  GLuint m_VAO;
  GLuint m_pointCloudCoordVBO, m_pointCloudColorVBO;
  GLuint m_boxVBO, m_boxEBO;
  GLuint m_gridPosVBO, m_gridColVBO, m_gridEBO;

  std::unique_ptr<OGLShader> m_pointCloudShader;
  std::unique_ptr<OGLShader> m_boxShader;
  std::unique_ptr<OGLShader> m_gridShader;

  size_t m_boxSize;
  size_t m_gridRes;
  size_t m_numDisplayedEntities;
  size_t m_numMaxEntities;

  bool m_isBoxVisible;
  bool m_isGridVisible;

  std::unique_ptr<Camera> m_camera;

  void* m_pointCloudCoordsBufferStart;
  void* m_pointCloudColorsBufferStart;

  typedef std::array<float, 3> Vertex;
  const std::array<Vertex, 8> m_refCubeVertices {
    Vertex({ 1.f, -1.f, -1.f }),
    Vertex({ 1.f, 1.f, -1.f }),
    Vertex({ -1.f, 1.f, -1.f }),
    Vertex({ -1.f, -1.f, -1.f }),
    Vertex({ 1.f, -1.f, 1.f }),
    Vertex({ 1.f, 1.f, 1.f }),
    Vertex({ -1.f, 1.f, 1.f }),
    Vertex({ -1.f, -1.f, 1.f })
  };

  const std::array<GLuint, 24> m_refCubeIndices {
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
};
}
