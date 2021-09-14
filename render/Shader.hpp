#pragma once

#include "Math.hpp"
#include <glad/glad.h>
#include <string>

namespace Render
{
class Shader
{
  public:
  Shader(const char* vert, const char* frag);
  ~Shader();

  void activate();
  void deactivate();

  // activate shader program before calling these functions
  void setUniform(const std::string& name, bool value) const {};
  void setUniform(const std::string& name, int value) const;
  void setUniform(const std::string& name, float value) const;
  void setUniform(const std::string& name, const Math::float4x4& mat) const;
  void setUniform(const std::string& name, const Math::float3& vec) const;

  const GLuint getProgramID() const { return m_programID; }

  private:
  GLint getUniformLocation(const std::string& name) const;

  void compileShader(GLenum type, const char* source);
  GLuint m_programID;
};
}
