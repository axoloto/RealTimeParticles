#include "OGLRender.hpp"
#include "GLSL.hpp"
#include "Math.hpp"
#include "imgui/imgui.h"

using namespace Render;

OGLRender::OGLRender() : m_mousePrevPos({0.0, 0.0})
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
    //vert1.xyz = { 100.f, 50.f, 4.f};
    vert2.xyz = { 0.2f, 0.1f, 0.1f};
    //vert2.xyz = { 0.2f, 20.f, 6.f};

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
    checkMouseEvents();

    updatePointCloud();

    m_pointCloudShader->activate();

    Math::float4x4 projViewMat = m_camera->getProjViewMat();
    //Math::float4x4 projViewMat = Math::float4x4::Identity();
    m_pointCloudShader->setUniform("u_projView", projViewMat);
    //m_pointCloudShader->setUniform("u_test", test);

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


void OGLRender::checkMouseEvents()
{
    ImGui::Begin("Navigation Pad");

    ImGui::Text("Touch me if you want to move around");

    if (!(ImGui::IsWindowHovered() && ImGui::IsWindowFocused())) 
    {
        ImGui::End();
        return;
    }

    // Zoom
    auto& io = ImGui::GetIO();
    if (io.MouseWheel != 0)
    {
        m_camera.get()->zoom(io.MouseWheel);
    }

    // Rotation
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) ) 
    {
        Math::float2 mousePos = Math::float2(ImGui::GetMousePos().x, ImGui::GetMousePos().y);

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
        {
            const auto delta = mousePos - m_mousePrevPos;
            const auto angle = delta * Math::PI_F / 180.0f * 0.5;
            m_camera.get()->rotate(-angle.y, angle.x);
        }

        m_mousePrevPos = mousePos;
    }

    // Translation
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ) 
    {
        Math::float2 mousePos = Math::float2(ImGui::GetMousePos().x, ImGui::GetMousePos().y);

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
        {
            const auto delta = mousePos - m_mousePrevPos;
            const auto displacement = -0.2f * delta;
            m_camera.get()->translate(displacement.x, displacement.y);
        }

        m_mousePrevPos = mousePos;
    }

    ImGui::End();
}