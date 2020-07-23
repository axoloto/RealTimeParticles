#pragma once 

#include <glad/glad.h>
#include "OGLShader.hpp"
#include <array>
#include <vector>
#include <memory>

namespace Render {
    class OGLRender {
        public:
            OGLRender();
            ~OGLRender();

            void draw();

        private:
            void buildShaders();
            void createPointCloudVBO();
            void createBoxVBO();
            void connectVBOsToVAO();
            void updatePointCloud();
            void drawPointCloud();
            void drawBox();

            const GLuint m_pointCloudAttribIndex { 0 }, m_boxAttribIndex { 1 };
            GLuint m_pointCloudVBO, m_boxVBO, m_VAO;

            std::unique_ptr<OGLShader> m_pointCloudShader;
            std::unique_ptr<OGLShader> m_boxShader;

            struct Vertex
            {
                std::array<float, 3> xyz;
            };

            std::vector<Vertex> m_pointCloudVertices;
    };
}
