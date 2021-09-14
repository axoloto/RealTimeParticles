
#include "Shader.hpp"
#include "GLSL.hpp"
#include "Logging.hpp"
#include <iostream>

using namespace Render;

Shader::Shader(const char* vert, const char* frag)
{
  m_programID = glCreateProgram();

  compileShader(GL_VERTEX_SHADER, vert);
  compileShader(GL_FRAGMENT_SHADER, frag);

  glLinkProgram(m_programID);

  GLint status;
  glGetProgramiv(m_programID, GL_LINK_STATUS, &status);

  if (status == GL_FALSE)
  {
    LOG_ERROR("Render: Shader linking failed");
    return;
  }
}

Shader::~Shader()
{
  if (m_programID != 0)
    glDeleteProgram(m_programID);
}

void Shader::compileShader(GLenum type, const char* source)
{
  GLuint shaderID = glCreateShader(type);

  glShaderSource(shaderID, 1, &source, nullptr);
  glCompileShader(shaderID);

  GLint status;
  glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);

  if (status == GL_FALSE)
  {
    LOG_ERROR("Render: Shader creation failed");
    return;
  }

  glAttachShader(m_programID, shaderID);
  glDeleteShader(shaderID);
}

void Shader::activate()
{
  glUseProgram(m_programID);
}

void Shader::deactivate()
{
  glUseProgram(0);
}

GLint Shader::getUniformLocation(const std::string& name) const
{
  return glGetUniformLocation(m_programID, name.c_str());
}

void Shader::setUniform(const std::string& name, int value) const
{
  glUniform1i(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, float value) const
{
  glUniform1f(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, const Math::float4x4& mat) const
{
  glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setUniform(const std::string& name, const Math::float3& vec) const
{
  glUniform3f(getUniformLocation(name), vec[0], vec[1], vec[2]);
}
