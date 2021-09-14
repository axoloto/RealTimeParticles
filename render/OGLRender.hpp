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

struct OGLRenderParams
{
  size_t currNbParticles = 0;
  size_t maxNbParticles = 0;
  size_t boxSize = 0;
  size_t gridRes = 0;
  float aspectRatio = 0.0f;
};

class OGLRender
{
  public:
  OGLRender(OGLRenderParams params);
  ~OGLRender();

  void checkMouseEvents(UserAction action, Math::float2 mouseDisplacement);
  void draw();

  inline const Math::float3 cameraPos() const { return m_camera->cameraPos(); }
  inline const Math::float3 focusPos() const { return m_camera->focusPos(); }

  inline void resetCamera() { m_camera->reset(); }
  inline void setWindowSize(Math::int2 windowSize)
  {
    if (m_camera)
      m_camera->setSceneAspectRatio((float)windowSize.x / windowSize.y);
  }
  inline void setNbParticles(int nbParticles) { m_nbParticles = nbParticles; }

  inline bool isBoxVisible() const { return m_isBoxVisible; }
  inline void setBoxVisibility(bool isVisible) { m_isBoxVisible = isVisible; }

  inline bool isGridVisible() const { return m_isGridVisible; }
  inline void setGridVisibility(bool isVisible) { m_isGridVisible = isVisible; }

  inline bool isTargetVisible() const { return m_isTargetVisible; }
  inline void setTargetVisibility(bool isVisible) { m_isTargetVisible = isVisible; }

  inline void setTargetPos(const Math::float3& pos) { m_targetPos = pos; }

  inline GLuint pointCloudCoordVBO() const { return m_pointCloudCoordVBO; }
  inline GLuint cameraCoordVBO() const { return m_cameraVBO; }
  inline GLuint gridDetectorVBO() const { return m_gridDetectorVBO; }

  private:
  void buildShaders();

  void initPointCloud();
  void drawPointCloud();

  void initBox();
  void drawBox();

  void initGrid();
  void drawGrid();

  void initTarget();
  void drawTarget();

  void loadCameraPos();

  void initCamera(float sceneAspectRatio);

  const GLuint m_pointCloudPosAttribIndex { 0 };
  const GLuint m_pointCloudColAttribIndex { 1 };
  const GLuint m_boxPosAttribIndex { 2 };
  const GLuint m_gridPosAttribIndex { 3 };
  const GLuint m_gridDetectorAttribIndex { 4 };
  const GLuint m_targetPosAttribIndex { 5 };

  GLuint m_VAO;
  GLuint m_pointCloudCoordVBO, m_pointCloudColorVBO;
  GLuint m_boxVBO, m_boxEBO;
  GLuint m_gridPosVBO, m_gridDetectorVBO, m_gridEBO;
  GLuint m_targetVBO;
  GLuint m_cameraVBO;

  std::unique_ptr<OGLShader> m_pointCloudShader;
  std::unique_ptr<OGLShader> m_boxShader;
  std::unique_ptr<OGLShader> m_gridShader;
  std::unique_ptr<OGLShader> m_targetShader;

  size_t m_boxSize;
  size_t m_gridRes;
  size_t m_nbParticles;
  size_t m_maxNbParticles;

  bool m_isBoxVisible;
  bool m_isGridVisible;
  bool m_isTargetVisible;

  Math::float3 m_targetPos;

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
