#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "Math.hpp"
#include "imgui/imgui.h"
#include <vector>

using namespace Render;

OGLRender::OGLRender() : m_halfboxSize(30)
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    initCamera();

    buildShaders();

    connectVBOsToVAO();

    int index = 0;
    for(int i = 0; i < 2; ++i)
    {
        int x = (2 * i - 1) * m_halfboxSize;
        for(int j = 0; j < 2; ++j)
        {
            int y = (2 * j - 1) * m_halfboxSize;
            for(int k = 0; k < 2; ++k)
            {
                int z = (2 * k - 1) * m_halfboxSize;
                // m_boxVertices[index].xyz = {(float) x, (float) y, (float) z};
                // m_boxVertices[index++].rgb = {1.f, 1.f, 1.f};
                Vertex vert;
                vert.xyz = {(float) x, (float) y, (float) z};
                vert.rgb = {1.f, 1.f, 1.f};
                m_pointCloudVertices.push_back(vert);
            }
        }
    }

    int halfboxSize = m_halfboxSize - 20;
    for(int i = 0; i < 2; ++i)
    {
        int x = (2 * i - 1) * halfboxSize;
        for(int j = 0; j < 2; ++j)
        {
            int y = (2 * j - 1) * halfboxSize;
            for(int k = 0; k < 2; ++k)
            {
                int z = (2 * k - 1) * halfboxSize;
                // m_boxVertices[index].xyz = {(float) x, (float) y, (float) z};
                // m_boxVertices[index++].rgb = {1.f, 1.f, 1.f};
                Vertex vert;
                vert.xyz = {(float) x, (float) y, (float) z};
                vert.rgb = {0.f, 0.f, 1.f};
                m_boxVertices.push_back(vert);
            }
        }
    }
}

OGLRender::~OGLRender()
{
    glDeleteBuffers(1, &m_pointCloudVBO);
    glDeleteBuffers(1, &m_boxVBO);
}

void OGLRender::initCamera()
{
    m_camera = std::make_unique<Camera>();
}

void OGLRender::buildShaders()
{
    m_boxShader = std::make_unique<OGLShader>(Render::VertShader, Render::FragShader);
    m_pointCloudShader = std::make_unique<OGLShader>(Render::VertShader, Render::FragShader);
}

void OGLRender::connectVBOsToVAO()
{
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glVertexAttribPointer(m_boxPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glEnableVertexAttribArray(m_boxPosAttribIndex);
    glVertexAttribPointer(m_boxColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(m_boxColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_pointCloudVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glVertexAttribPointer(m_pointCloudPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
    glVertexAttribPointer(m_pointCloudColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(m_pointCloudColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OGLRender::draw()
{
    //drawPointCloud();
    drawBox();
}

void OGLRender::updatePointCloud()
{
    if(m_pointCloudVertices.empty()) return;

    size_t pointCloudSize = sizeof(m_pointCloudVertices[0]) * m_pointCloudVertices.size();

    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glBufferData(GL_ARRAY_BUFFER, pointCloudSize, &m_pointCloudVertices[0], GL_STREAM_DRAW);
}

void OGLRender::drawPointCloud()
{
    updatePointCloud();

    m_pointCloudShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_pointCloudShader->setUniform("u_projView", projViewMat);

    glDrawArrays(GL_POINTS, 0, (GLsizei) m_pointCloudVertices.size());

    m_pointCloudShader->deactivate();
}

void OGLRender::updateBox()
{
    size_t boxSize = sizeof(m_boxVertices[0]) * m_boxVertices.size();

    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glBufferData(GL_ARRAY_BUFFER, boxSize, &m_boxVertices[0], GL_STREAM_DRAW);
}

void OGLRender::drawBox()
{
    updateBox();

    m_boxShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_boxShader->setUniform("u_projView", projViewMat);

    glDrawArrays(GL_POINTS, 0, (GLsizei) m_boxVertices.size());

    m_boxShader->deactivate();
}

void OGLRender::checkMouseEvents(UserAction action, Math::float2 delta)
{
    switch(action)
    {
        case UserAction::TRANSLATION :
        {
            const auto displacement = 0.2f * delta;
            m_camera.get()->translate(-displacement.x, displacement.y);
            break;
        }
        case UserAction::ROTATION :
        {
            const auto angle = delta * Math::PI_F / 180.0f * 0.5;
            m_camera.get()->rotate(angle.x, angle.y);
            break;
        }
        case UserAction::ZOOM :
        {
            m_camera.get()->zoom(delta.x);
            break;
        }
    }
}