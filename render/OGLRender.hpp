#pragma once 

#include <glad/glad.h>
#include "OGLShader.hpp"
#include "Camera.hpp"
#include <array>
#include <vector>
#include <memory>


namespace Render {

    enum class UserAction { TRANSLATION, ROTATION, ZOOM };

    class OGLRender {
        public:
            OGLRender();
            ~OGLRender();

            void checkMouseEvents(UserAction action, Math::float2 mouseDisplacement);
            void draw();

            inline const Math::float3 cameraPos() const { return m_camera->cameraPos();}
            inline const Math::float3 targetPos() const { return m_camera->targetPos();}

            inline void resetCamera() { m_camera->reset(); }

        private:

            void buildShaders();
            void connectVBOsToVAO();
            void generateBox();
            void generatePointCloudVBO();

            void updatePointCloud();
            void drawPointCloud();

            void drawBox();

            void initCamera();

            const GLuint m_pointCloudPosAttribIndex { 0 }, m_pointCloudColAttribIndex { 1 }, m_boxPosAttribIndex { 2 }, m_boxColAttribIndex { 3 };
            GLuint m_pointCloudVBO, m_boxVBO, m_boxEBO, m_VAO;

            std::unique_ptr<OGLShader> m_pointCloudShader;
            std::unique_ptr<OGLShader> m_boxShader;

            struct Vertex
            {
                std::array<float, 3> xyz;
                std::array<float, 3> rgb;
            };

            std::vector<Vertex> m_pointCloudVertices;

            int m_halfboxSize;

            std::unique_ptr<Camera> m_camera;
    };
}
