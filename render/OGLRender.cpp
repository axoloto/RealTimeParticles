#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "imgui/imgui.h"
#include "diligentGraphics/Math.hpp"

using namespace Render;

OGLRender::OGLRender(int boxSize, int numEntities, float sceneAspectRatio) : m_boxSize(boxSize), m_numEntities(numEntities)
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    initCamera(sceneAspectRatio);

    buildShaders();

    connectVBOsToVAO();

    generateBox();
}

OGLRender::~OGLRender()
{
    glDeleteBuffers(1, &m_pointCloudCoordVBO);
    glDeleteBuffers(1, &m_pointCloudColorVBO);
    glDeleteBuffers(1, &m_boxVBO);
}

void OGLRender::initCamera(float sceneAspectRatio)
{
    m_camera = std::make_unique<Camera>(sceneAspectRatio);
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

    glGenBuffers(1, &m_pointCloudCoordVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudCoordVBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * 100000 * sizeof(float), NULL, GL_DYNAMIC_DRAW); // WIP
    glVertexAttribPointer(m_pointCloudPosAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
    glEnableVertexAttribArray(m_pointCloudPosAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &m_pointCloudColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudColorVBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * 100000 * sizeof(float), NULL, GL_DYNAMIC_DRAW); // WIP
    glVertexAttribPointer(m_pointCloudColAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
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

void OGLRender::setPointCloudBuffers(void* coordsBufferStart, void* colorsBufferStart)
{ 
    m_pointCloudCoordsBufferStart = coordsBufferStart;
    m_pointCloudColorsBufferStart = colorsBufferStart;
}

void OGLRender::updatePointCloud()
{
    if(m_numEntities > 0)
    {
        size_t vertBufferSize = 4 * sizeof(float) * m_numEntities;
        glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudCoordVBO);
        glBufferData(GL_ARRAY_BUFFER, vertBufferSize, m_pointCloudCoordsBufferStart, GL_DYNAMIC_DRAW);

        size_t colBufferSize = 4 * sizeof(float) * m_numEntities;
        glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudColorVBO);
        glBufferData(GL_ARRAY_BUFFER, colBufferSize, m_pointCloudColorsBufferStart, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void OGLRender::drawPointCloud()
{
    //updatePointCloud();

    m_pointCloudShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    m_pointCloudShader->setUniform("u_projView", projViewMat);

    glDrawArrays(GL_POINTS, 0, (GLsizei) m_numEntities);

    m_pointCloudShader->deactivate();
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

    int boxHalfSize = m_boxSize / 2;
    for(auto& vertex : boxVertices)
    {
        float x = vertex.xyz[0] * boxHalfSize;
        float y = vertex.xyz[1] * boxHalfSize;
        float z = vertex.xyz[2] * boxHalfSize;
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
            m_camera->translate(-displacement.x, displacement.y);
            break;
        }
        case UserAction::ROTATION :
        {
            const auto angle = delta * Math::PI_F / 180.0f * 0.5;
            m_camera->rotate(angle.y, angle.x);
            break;
        }
        case UserAction::ZOOM :
        {
            m_camera->zoom(delta.x);
            break;
        }
    }
}