#pragma once 

#include "Math.hpp"

namespace Render {
    class Camera {
    public:
        Camera(float sceneAspectRatio);
        ~Camera()=default;

        void reset(); 

        inline void setSceneAspectRatio(float aspectRatio) { m_aspectRatio = aspectRatio; updateProjMat(); updateProjViewMat(); }

        void rotate(float angleX, float angleY);
        void translate(float dispX, float dispY);
        void zoom(float delta);

        inline const Math::float3 cameraPos() const { return m_cameraPos; }
        inline const Math::float3 targetPos() const { return m_targetPos; }

        inline Math::float4x4 getProjViewMat() const { return m_projViewMat; }
    private:
        void updateProjMat();
        void updateProjViewMat();

        Math::float3 m_cameraPos, m_cameraInitPos;
        Math::float3 m_targetPos, m_targetInitPos;
        Math::float4x4 m_projMat, m_viewMat, m_projViewMat;

        float m_fov;
        float m_aspectRatio;
        float m_zNear;
        float m_zFar;
    };
}

