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
  OGLRender(int halfBoxSize, int numDisplayedEntities, int numMaxEntities, float aspectRatio);
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

  private:
  void buildShaders();
  void connectVBOsToVAO();

  void drawPointCloud();

  void generateBox();
  void drawBox();

  void generateGrid();
  void drawGrid();

  void initCamera(float sceneAspectRatio);

  const GLuint
      m_pointCloudPosAttribIndex { 0 },
      m_pointCloudColAttribIndex { 1 },
      m_boxPosAttribIndex { 2 }, m_boxColAttribIndex { 3 },
      m_gridPosAttribIndex { 4 }, m_gridColAttribIndex { 5 };

  GLuint m_VAO;
  GLuint m_pointCloudCoordVBO, m_pointCloudColorVBO;
  GLuint m_boxVBO, m_boxEBO;
  GLuint m_gridVBO, m_gridEBO;

  std::unique_ptr<OGLShader> m_pointCloudShader;
  std::unique_ptr<OGLShader> m_boxShader;
  std::unique_ptr<OGLShader> m_gridShader;

  int m_boxSize;
  int m_numDisplayedEntities;
  int m_numMaxEntities;

  bool m_isBoxVisible;
  bool m_isGridVisible;

  std::unique_ptr<Camera> m_camera;

  void* m_pointCloudCoordsBufferStart;
  void* m_pointCloudColorsBufferStart;
};
}
