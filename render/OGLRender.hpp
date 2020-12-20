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

  void setPointCloudBuffers(void* coordsBufferStart, void* colorsBufferStart);
  GLuint pointCloudCoordVBO() { return m_pointCloudCoordVBO; }
  GLuint pointCloudColorVBO() { return m_pointCloudColorVBO; }

  private:
  void buildShaders();
  void connectVBOsToVAO();
  void generateBox();

  void drawPointCloud();

  void drawBox();

  void initCamera(float sceneAspectRatio);

  const GLuint m_pointCloudPosAttribIndex { 0 }, m_pointCloudColAttribIndex { 1 }, m_boxPosAttribIndex { 2 }, m_boxColAttribIndex { 3 };
  GLuint m_pointCloudCoordVBO, m_pointCloudColorVBO, m_boxVBO, m_boxEBO, m_VAO;

  std::unique_ptr<OGLShader> m_pointCloudShader;
  std::unique_ptr<OGLShader> m_boxShader;

  int m_boxSize;
  int m_numDisplayedEntities;
  int m_numMaxEntities;

  std::unique_ptr<Camera> m_camera;

  void* m_pointCloudCoordsBufferStart;
  void* m_pointCloudColorsBufferStart;
};
}
