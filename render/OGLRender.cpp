#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "Math.hpp"
#include "imgui/imgui.h"
#include <vector>
#include <stdlib.h> 

using namespace Render;

OGLRender::OGLRender(int halfBoxSize, int numEntities) : m_halfboxSize(halfBoxSize), m_numEntities(numEntities)
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    initCamera();

    buildShaders();

    connectVBOsToVAO();

    generateBox();
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

void OGLRender::generateBox()
{
    struct Vertex
    {
        std::array<float, 3> xyz;
        std::array<float, 3> rgb;
    };

    // VBO
    int index = 0;
    std::array<Vertex, 8> boxVertices;
    boxVertices[0].xyz = { 1.f, -1.f, -1.f};
    boxVertices[1].xyz = { 1.f,  1.f, -1.f};
    boxVertices[2].xyz = {-1.f,  1.f, -1.f};
    boxVertices[3].xyz = {-1.f, -1.f, -1.f};
    boxVertices[4].xyz = { 1.f, -1.f,  1.f};
    boxVertices[5].xyz = { 1.f,  1.f,  1.f};
    boxVertices[6].xyz = {-1.f,  1.f,  1.f};
    boxVertices[7].xyz = {-1.f, -1.f,  1.f};

    for(auto& vertex : boxVertices)
    {
        float x = vertex.xyz[0] * m_halfboxSize;
        float y = vertex.xyz[1] * m_halfboxSize;
        float z = vertex.xyz[2] * m_halfboxSize;
        vertex.xyz = {x, y, z};
        vertex.rgb = {1.f, 1.f, 1.f};
    }

    size_t boxBufferSize = sizeof(boxVertices[0]) * boxVertices.size();
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glBufferData(GL_ARRAY_BUFFER, boxBufferSize, &boxVertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // EBO
    GLuint boxIndices[] = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 7,
        7, 4,
        0, 4,
        1, 5,
        2, 6,
        3, 7    
    };

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(boxIndices), boxIndices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OGLRender::buildShaders()
{
    m_pointCloudShader = std::make_unique<OGLShader>(Render::PointCloudVertShader, Render::FragShader);
    m_boxShader = std::make_unique<OGLShader>(Render::BoxVertShader, Render::FragShader);
}

void OGLRender::connectVBOsToVAO()
{
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_pointCloudVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glVertexAttribPointer(m_pointCloudPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
    glVertexAttribPointer(m_pointCloudColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(m_pointCloudColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glVertexAttribPointer(m_boxPosAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), NULL);
    glEnableVertexAttribArray(m_boxPosAttribIndex);
    glVertexAttribPointer(m_boxColAttribIndex, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(m_boxColAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_boxEBO);
}

void OGLRender::draw()
{
    drawPointCloud();
    drawBox();
}

void OGLRender::updatePointCloud()
{
    if(m_pointCloudBufferSize > 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
        glBufferData(GL_ARRAY_BUFFER, m_pointCloudBufferSize, m_pointCloudBufferStart, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void OGLRender::drawPointCloud()
{
    updatePointCloud();

    m_pointCloudShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_pointCloudShader->setUniform("u_projView", projViewMat);

    glDrawArrays(GL_POINTS, 0, (GLsizei) m_numEntities);

    m_pointCloudShader->deactivate();
}

void OGLRender::drawBox()
{
    m_boxShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_boxShader->setUniform("u_projView", projViewMat);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEBO);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_boxShader->deactivate();
}

void OGLRender::checkMouseEvents(UserAction action, Math::float2 delta)
{
    switch(action)
    {
        case UserAction::TRANSLATION :
        {
            const auto displacement = 0.4f * delta;
            m_camera.get()->translate(-displacement.x, displacement.y);
            break;
        }
        case UserAction::ROTATION :
        {
            const auto angle = delta * Math::PI_F / 180.0f * 0.5;
            m_camera.get()->rotate(angle.y, angle.x);
            break;
        }
        case UserAction::ZOOM :
        {
            m_camera.get()->zoom(delta.x);
            break;
        }
    }
}