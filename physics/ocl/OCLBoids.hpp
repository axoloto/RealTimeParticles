#pragma once

#include <array>
#include "Boids.hpp"
#include "CL/cl.h"

namespace Core
{

    class OCLBoids : public Boids
    {
    public:
        OCLBoids(int boxSize, int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
        ~OCLBoids();

        void updatePhysics() override;

    private: 
        bool initOpenCL();
        bool createRandomKernel(unsigned int bufferGL);
        bool createColorKernel(unsigned int bufferGL);
        void runKernel(cl_kernel kernel, cl_mem GLCLbuffer);

        bool m_init;

        cl_platform_id cl_platform;
        cl_device_id cl_device;
        cl_context cl_context;
        cl_program cl_program;
        cl_command_queue cl_queue;

        cl_kernel cl_randomKernel;
        cl_mem cl_randomBuff;

        cl_kernel cl_colorKernel;
        cl_mem cl_colorBuff;
    };
}