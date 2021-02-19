
#pragma once

namespace Render
{
constexpr char PointCloudVertShader[] = R"(#version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aCol;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(aCol, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);

        vec4 eye = u_projView * vec4(aPos, 1.0); 
        float d = length(eye);
        gl_PointSize = max(3000.0 * 1.0/(0.04 + 0.8*d + 0.0002*d*d), 0.8); 			
    }
    )";

constexpr char PointCloudFragShader[] = R"(#version 330 core
    in vec4 vertexColor;

    out vec4 fragColor;

    void main()
    {
      // Alpha blending
      vec2 xy = 2.0 * gl_PointCoord.xy - vec2(1.0);
      float r2 = dot(xy, xy);
      fragColor.a = exp2(-r2 * 0.2);
      //fragColor.a = 1.0;

      fragColor.rgb = vertexColor.rgb * fragColor.a;
      //fragColor = vertexColor;
    }
    )";

constexpr char BoxVertShader[] = R"(#version 330 core
    layout(location = 2) in vec3 aPos;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);
        gl_PointSize = 4.0;
    }
    )";

constexpr char GridVertShader[] = R"(#version 330 core
    layout(location = 3) in vec3 aPos;
    layout(location = 4) in float vertexDetector;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
      vertexColor = vec4(0.6, 0.6, 0.2, 0.6);
      
      if(vertexDetector > 0.0f)
        gl_Position = u_projView * vec4(aPos, 1.0);
      else
        gl_Position =  vec4(0.0, 0.0, 0.0, 1.0);

      gl_PointSize = 1.0;
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