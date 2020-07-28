#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "Math.hpp"
#include "imgui/imgui.h"

using namespace Render;

OGLRender::OGLRender() : m_halfboxSize(30)
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    initCamera();

    buildShaders();

    createPointCloudVBO();
    createBoxVBO();

    connectVBOsToVAO();

    Vertex vert1, vert2;
    vert1.xyz = { 0.1f, 0.4f, 0.2f};
    vert1.rgb = { 0.1f, 0.4f, 0.2f};
    //vert1.xyz = { 100.f, 50.f, 4.f};
    vert2.xyz = { 0.2f, 0.1f, 0.1f};
    vert2.rgb = { 0.9f, 0.4f, 0.2f};

    m_pointCloudVertices.push_back(vert1);
    m_pointCloudVertices.push_back(vert2);
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
    m_pointCloudShader = std::make_unique<OGLShader>(Render::VertShader, Render::FragShader);
    m_boxShader = std::make_unique<OGLShader>(Render::VertShader, Render::FragShader);
}

void OGLRender::createPointCloudVBO()
{
    glGenBuffers(1, &m_pointCloudVBO);
}

void OGLRender::createBoxVBO()
{
    glGenBuffers(1, &m_boxVBO);

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
                m_boxVertices[index].xyz = {(float) x, (float) y, (float) z};
                m_boxVertices[index++].rgb = {1.0/255.0f, 1.0/255.0f, 1.0/255.0f};
            }
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glBufferData(GL_ARRAY_BUFFER, 8, &m_boxVertices[0], GL_STATIC_DRAW);
}

void OGLRender::connectVBOsToVAO()
{
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
    glEnableVertexAttribArray(m_pointCloudColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glVertexAttribPointer(m_pointCloudPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glVertexAttribPointer(m_pointCloudColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnableVertexAttribArray(m_boxPosAttribIndex);
    glEnableVertexAttribArray(m_boxColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glVertexAttribPointer(m_boxPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glVertexAttribPointer(m_boxColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glDisableVertexAttribArray(m_VAO);
}

void OGLRender::draw()
{
    drawPointCloud();
    drawBox();
}

void OGLRender::drawPointCloud()
{
    updatePointCloud();

    m_pointCloudShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_pointCloudShader->setUniform("u_projView", projViewMat);

    glBindVertexArray(m_VAO);
    glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
    glEnableVertexAttribArray(m_pointCloudColAttribIndex);

    glDrawArrays(GL_POINTS, 0, (GLsizei) m_pointCloudVertices.size());

    glDisableVertexAttribArray(m_VAO);

    m_pointCloudShader->deactivate();
}

void OGLRender::updatePointCloud()
{
    size_t pointCloudSize = sizeof(m_pointCloudVertices[0]) * m_pointCloudVertices.size();

    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glBufferData(GL_ARRAY_BUFFER, pointCloudSize, &m_pointCloudVertices[0], GL_STREAM_DRAW);
}

void OGLRender::drawBox()
{
    m_boxShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_boxShader->setUniform("u_projView", projViewMat);

    glBindVertexArray(m_VAO);
    glEnableVertexAttribArray(m_boxPosAttribIndex);
    glEnableVertexAttribArray(m_boxColAttribIndex);

    glDrawArrays(GL_POINTS, 0, 8);

    glDisableVertexAttribArray(m_VAO);

    m_boxShader->deactivate();
}

void OGLRender::checkMouseEvents(UserAction action, Math::int2 delta)
{
    Math::float2 fDelta((float) delta.x, (float) delta.y);

    switch(action)
    {
        case UserAction::TRANSLATION :
        {
            const auto displacement = 0.2f * fDelta;
            m_camera.get()->translate(-displacement.x, displacement.y);
            break;
        }
        case UserAction::ROTATION :
        {
            const auto angle = fDelta * Math::PI_F / 180.0f * 0.5;
            m_camera.get()->rotate(angle.x, angle.y);
            break;
        }
        case UserAction::ZOOM :
        {
            m_camera.get()->zoom(fDelta.x);
            break;
        }
    }
}