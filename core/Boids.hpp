#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Boids : public Physics {
        public:
            Boids(int boxSize, int numEntities);
            ~Boids() = default;

            void updatePhysics() override;
            void setSteeringMaxForce(float steeringMaxForce) { m_steeringMaxForce = steeringMaxForce; }
            void setmaxVelocity(float maxVelocity) { m_maxVelocity = maxVelocity; }

        private:
            void generateBoids();
            Math::float3 target;
            float time =0;
            float m_steeringMaxForce;
            float m_maxVelocity;
    };
}

