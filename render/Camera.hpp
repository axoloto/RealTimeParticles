#pragma once 

#include "Math.hpp"

namespace Render {
    class Camera {
    public:
        Camera();
        ~Camera()=default;

        void rotate(float angleX, float angleY);
        void translate(float dispX, float dispY);
        void zoom(float delta);

        inline Math::float4x4 getProjViewMat() const { return m_projViewMat; }
        
    private:
        void updateProjViewMat();

        Math::float3 m_cameraPos, m_targetPos;
        Math::float4x4 m_projMat, m_viewMat, m_projViewMat;

        float m_fov;
        float m_aspectRatio;
        float m_zNear;
        float m_zFar;
    };
}

