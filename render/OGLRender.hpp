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
            OGLRender(int halfBoxSize, int numEntities, float aspectRatio);
            ~OGLRender();

            void checkMouseEvents(UserAction action, Math::float2 mouseDisplacement);
            void draw();

            inline const Math::float3 cameraPos() const { return m_camera->cameraPos();}
            inline const Math::float3 targetPos() const { return m_camera->targetPos();}

            inline void resetCamera() { m_camera->reset(); }

            inline void setPointCloudBuffer(void* bufferStart, size_t bufferSize) { m_pointCloudBufferStart = bufferStart; m_pointCloudBufferSize = bufferSize; }
            inline void setWindowSize(Math::int2 windowSize) { if(m_camera) m_camera->setSceneAspectRatio((float) windowSize.x / windowSize.y); }
        private:

            void buildShaders();
            void connectVBOsToVAO();
            void generateBox();

            void updatePointCloud();
            void drawPointCloud();

            void drawBox();

            void initCamera(float sceneAspectRatio);

            const GLuint m_pointCloudPosAttribIndex { 0 }, m_pointCloudColAttribIndex { 1 }, m_boxPosAttribIndex { 2 }, m_boxColAttribIndex { 3 };
            GLuint m_pointCloudVBO, m_boxVBO, m_boxEBO, m_VAO;

            std::unique_ptr<OGLShader> m_pointCloudShader;
            std::unique_ptr<OGLShader> m_boxShader;

            int m_halfboxSize;
            int m_numEntities;

            std::unique_ptr<Camera> m_camera;

            // WIP
            size_t m_pointCloudBufferSize; 
            void* m_pointCloudBufferStart;
    };
}
