#pragma once

#include <array>
#include "Boids.hpp"

namespace Core
{

    class OCLBoids : public Boids
    {
    public:
        OCLBoids(int boxSize, int numEntities);
        ~OCLBoids() = default;

        //void updatePhysics() override;

    private: 
        int initOpenCL();
    };
}