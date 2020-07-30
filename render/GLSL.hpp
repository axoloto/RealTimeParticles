
#pragma once

namespace Render {

    constexpr char VertPointCloudShader[] = R"(#version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aCol;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(aCol, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);
        gl_PointSize = 10.0;
    }
    )";

    constexpr char VertBoxShader[] = R"(#version 330 core
    layout(location = 2) in vec3 aPos;
    layout(location = 3) in vec3 aCol;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(aCol, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);
        gl_PointSize = 10.0;
    }
    )";

    constexpr char FragShader[] = R"(#version 330 core
    in vec4 vertexColor;

    out vec4 fragColor;

    void main()
    {
        fragColor = vertexColor;
    }
    )";

}