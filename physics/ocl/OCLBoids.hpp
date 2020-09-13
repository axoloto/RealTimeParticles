#pragma once

#include <array>
#include "Boids.hpp"
#include "CL/cl.h"

namespace Core
{

    class OCLBoids : public Boids
    {
    public:
        OCLBoids(int boxSize, int numEntities);
        ~OCLBoids();

        void updatePhysics() override;

    private: 
        bool initOpenCL();
        bool createKernel();
        void runKernel();

        bool m_init;

        cl_platform_id cl_platform;
        cl_device_id cl_device;
        cl_context cl_context;
        cl_program cl_program;
        cl_command_queue cl_queue;
        cl_kernel cl_boidKernel;

        cl_mem cl_res_buff;
    };
}