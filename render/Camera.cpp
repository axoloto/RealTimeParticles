#include "Camera.hpp"

using namespace Render;
using namespace Math;

Camera::Camera(float sceneAspectRatio)
    : m_fov(120.0f)
    , m_aspectRatio(sceneAspectRatio)
    , m_zNear(0.01f)
    , m_zFar(150000.f)
    , m_cameraInitPos({ 3600.0, 50.0, 0.0 })
    , m_targetInitPos({ 0.0, 0.0, 0.0 })
{
  reset();
}

void Camera::reset()
{
  m_cameraPos = m_cameraInitPos;
  m_targetPos = m_targetInitPos;
  updateProjMat();
  updateProjViewMat();
}

void Camera::rotate(float angleX, float angleY)
{
  float3 vecTargetCamera(m_cameraPos - m_targetPos);

  auto rot = float4x4::RotationY(angleY) * float4x4::RotationX(angleX);

  const auto matWorldToCam = m_viewMat.RemoveTranslation();
  rot = matWorldToCam * rot * matWorldToCam.Transpose();

  vecTargetCamera = vecTargetCamera * rot;
  m_cameraPos = m_targetPos + vecTargetCamera;

  updateProjViewMat();
}

void Camera::translate(float dispX, float dispY)
{
  auto trans = float4x4::Translation(dispX, dispY, 0.0f);

  const auto matWorldToCam = m_viewMat.RemoveTranslation();
  trans = matWorldToCam * trans * matWorldToCam.Transpose();

  m_cameraPos = float4(m_cameraPos, 1.0) * trans;
  m_targetPos = float4(m_targetPos, 1.0) * trans;

  updateProjViewMat();
}

void Camera::zoom(float delta)
{
  float zoomRatio = 1.f + delta / 10.f;
  float3 vecTargetCamera(m_cameraPos - m_targetPos);

  m_cameraPos = m_targetPos + vecTargetCamera * zoomRatio;

  updateProjViewMat();
}

void Camera::updateProjMat()
{
  m_projMat = float4x4::Projection(m_fov, m_aspectRatio, m_zNear, m_zFar, true);
}

void Camera::updateProjViewMat()
{
  // Classic lookAt function

  float3 refX, refY, refZ;

  refZ = -(m_cameraPos - m_targetPos);
  refZ = normalize(refZ);

  refY = { 0.0f, 1.0f, 0.0f };
  refX = cross(refY, refZ);
  refY = cross(refZ, refX);

  refX = normalize(refX);
  refY = normalize(refY);

  float dotRefXEye = -dot(refX, m_cameraPos);
  float dotRefYEye = -dot(refY, m_cameraPos);
  float dotRefZEye = -dot(refZ, m_cameraPos);

  m_viewMat = {
    refX.x, refY.x, refZ.x, 0.0f,
    refX.y, refY.y, refZ.y, 0.0f,
    refX.z, refY.z, refZ.z, 0.0f,
    dotRefXEye, dotRefYEye, dotRefZEye, 1.0f
  };

  m_projViewMat = float4x4::Mul(m_viewMat, m_projMat);
}
