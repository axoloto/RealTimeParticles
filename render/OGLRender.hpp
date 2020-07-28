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

            void checkMouseEvents(UserAction action, Math::int2 mouseDisplacement);
            void draw();

        private:

            void buildShaders();
            void createPointCloudVBO();
            void createBoxVBO();
            void connectVBOsToVAO();
            void updatePointCloud();
            void drawPointCloud();
            void drawBox();
            void initCamera();

            const GLuint m_pointCloudPosAttribIndex { 0 }, m_pointCloudColAttribIndex { 1 }, m_boxPosAttribIndex { 2 }, m_boxColAttribIndex { 3 };
            GLuint m_pointCloudVBO, m_boxVBO, m_VAO;

            std::unique_ptr<OGLShader> m_pointCloudShader;
            std::unique_ptr<OGLShader> m_boxShader;

            struct Vertex
            {
                std::array<float, 3> xyz;
                std::array<float, 3> rgb;
            };

            std::vector<Vertex> m_pointCloudVertices;

            std::array<Vertex, 8> m_boxVertices;
            int m_halfboxSize;

            std::unique_ptr<Camera> m_camera;
    };
}
