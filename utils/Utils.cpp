
#include "Utils.hpp"

#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>

const std::string Utils::GetSrcDir()
{
  return std::string(SOURCE_DIR);
}

const std::string Utils::GetInstallDir()
{
  return std::string(INSTALL_DIR);
}

const std::string Utils::GetVersions()
{
  return std::string(VERSION_MAJOR) + "." + std::string(VERSION_MINOR);
}

const std::string Utils::FloatToStr(float val, size_t precision)
{
  std::ostringstream str;
  str << std::fixed << std::setprecision(precision) << std::setfill('0') << val << "f";
  return str.str();
}