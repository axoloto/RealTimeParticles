
#pragma once

namespace Render {

    constexpr char VertShader[] = R"(#version 330 core
    layout(location = 0) in vec3 aPos;

    uniform mat4 u_projView;
    uniform float u_test;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
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