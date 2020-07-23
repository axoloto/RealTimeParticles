#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "Math.hpp"

using namespace Render;

OGLRender::OGLRender()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    buildShaders();

    createPointCloudVBO();
    createBoxVBO();

    connectVBOsToVAO();

    Vertex vert;
    vert.xyz = { 0.10f, 0.2f, -1.0f};

    m_pointCloudVertices.push_back(vert);
}

OGLRender::~OGLRender()
{
    glDeleteBuffers(1, &m_pointCloudVBO);
    glDeleteBuffers(1, &m_boxVBO);
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
}

void OGLRender::connectVBOsToVAO()
{
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glEnableVertexAttribArray(m_pointCloudAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glVertexAttribPointer(m_pointCloudAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnableVertexAttribArray(m_boxAttribIndex);
    glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
    glVertexAttribPointer(m_boxAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
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

    Math::float4x4 projViewMat = Math::float4x4::Identity();
    m_pointCloudShader->activate();
    m_pointCloudShader->setUniform("u_projView", projViewMat);

    glBindVertexArray(m_VAO);
    glEnableVertexAttribArray(m_pointCloudAttribIndex);

    glDrawArrays(GL_POINTS, 0, m_pointCloudVertices.size());

    glDisableVertexAttribArray(m_VAO);

    m_pointCloudShader->deactivate();
}

void OGLRender::updatePointCloud()
{
    int pointCloudSize = sizeof(m_pointCloudVertices[0]) * m_pointCloudVertices.size();

    glBindBuffer(GL_ARRAY_BUFFER, m_pointCloudVBO);
    glBufferData(GL_ARRAY_BUFFER, pointCloudSize, &m_pointCloudVertices[0], GL_STREAM_DRAW);
}

void OGLRender::drawBox()
{
    m_boxShader->activate();

    m_boxShader->deactivate();
}