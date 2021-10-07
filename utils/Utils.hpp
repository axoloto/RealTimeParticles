#pragma once

#include <string>

namespace Utils
{
const std::string GetSrcDir();
const std::string GetInstallDir();
const std::string GetVersions();
const std::string FloatToStr(float val, size_t precision = 10);
}
