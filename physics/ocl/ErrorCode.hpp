
#pragma once

#include "CL/cl2.hpp"
#include "Logging.hpp"
#include <string>

#define CL_ERROR(errorCode, errorStr) LOG_ERROR("OpenCL error at line {}:  {} {}", __LINE__, Physics::CL::ErrorCodeToStr(errorCode), errorStr)

namespace Physics::CL
{
std::string ErrorCodeToStr(cl_int error);
}