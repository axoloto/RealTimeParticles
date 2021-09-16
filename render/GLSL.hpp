
#pragma once

namespace Render
{
constexpr char PointCloudVertShader[] = R"(#version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aCol;

    uniform int u_pointSize;
    uniform mat4 u_projView;
    out vec4 vertexPos;

    void main()
    {
        vertexPos = vec4(aPos, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);

        vec4 eye = u_projView * vec4(aPos, 1.0); 
        float d = length(eye);
        // WIP
        gl_PointSize = u_pointSize ; 			
        //gl_PointSize = u_pointSize * max(3000.0 * 1.0/(0.04 + 0.8*d + 0.0002*d*d), 0.8); 			
    }
    )";

constexpr char PointCloudFragShader[] = R"(#version 330 core
    in vec4 vertexPos;

    uniform vec3 u_cameraPos;

    out vec4 fragColor;

    void main()
    {
      // Additive alpha blending
      vec3 xyz = u_cameraPos.xyz - vertexPos.xyz;

      // WIP not working well
      // Alpha tending to 1 close from the camera
      // to see non translucent close neighbor particles
      // And down to 0.75 away from the camera to allow additive blending
      float r2 = dot(xyz, xyz);
      fragColor.a = 2.5* exp(-r2 / 100000)+0.75;

      fragColor.rgb = vec3(0.8, 0.0, 0.0) * fragColor.a;
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

constexpr char TargetVertShader[] = R"(#version 330 core
    layout(location = 5) in vec3 aPos;

    uniform mat4 u_projView;
    out vec4 vertexColor;

    void main()
    {
        vertexColor = vec4(1.0, 1.0, 0.0, 1.0);
        gl_Position = u_projView * vec4(aPos, 1.0);

        vec4 eye = u_projView * vec4(aPos, 1.0); 
        float d = length(eye);
        gl_PointSize = 5 * max(3000.0 * 1.0/(0.04 + 0.8*d + 0.0002*d*d), 0.8); 			
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