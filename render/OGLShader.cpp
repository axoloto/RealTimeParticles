
#include "GLSL.hpp"
#include "OGLShader.hpp"
#include <iostream>

using namespace Render;

OGLShader::OGLShader(const char* vert, const char* frag)
{
  m_programID = glCreateProgram();

  compileShader(GL_VERTEX_SHADER, vert);
  compileShader(GL_FRAGMENT_SHADER, frag);

  glLinkProgram(m_programID);

  GLint status;
  glGetProgramiv(m_programID, GL_LINK_STATUS, &status);

  if (status == GL_FALSE)
  {
    std::cout << "shader linking failed" << std::endl;
    return;
  }
}

OGLShader::~OGLShader()
{
  if (m_programID != 0)
    glDeleteProgram(m_programID);
}

void OGLShader::compileShader(GLenum type, const char* source)
{
  GLuint shaderID = glCreateShader(type);

  glShaderSource(shaderID, 1, &source, nullptr);
  glCompileShader(shaderID);

  GLint status;
  glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);

  if (status == GL_FALSE)
  {
    std::cout << "shader creation failed" << std::endl;
    return;
  }

  glAttachShader(m_programID, shaderID);
  glDeleteShader(shaderID);
}

void OGLShader::activate()
{
  glUseProgram(m_programID);
}

void OGLShader::deactivate()
{
  glUseProgram(0);
}

GLint OGLShader::getUniformLocation(const std::string& name) const
{
  return glGetUniformLocation(m_programID, name.c_str());
}

void OGLShader::setUniform(const std::string& name, float value) const
{
  glUniform1f(getUniformLocation(name), value);
}

void OGLShader::setUniform(const std::string& name, const Math::float4x4& mat) const
{
  glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}
