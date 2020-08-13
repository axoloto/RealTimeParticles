#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Boids : public Physics {
        public:
            Boids(int boxSize, int numEntities);
            ~Boids() = default;

            void updatePhysics() override;

        private:
            void generateBoids();
    };
}


