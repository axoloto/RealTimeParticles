
#include "Utils.hpp"

#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>


const std::string Utils::GetSrcDir()
{
  return std::string(BINARY_DIR);
}

const std::string Utils::FloatToStr(float val, size_t precision)
{
  std::ostringstream str;
  str << std::fixed << std::setprecision(precision) << std::setfill('0') << val << "f";
  return str.str();
}