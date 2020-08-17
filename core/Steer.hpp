#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Steer : public Physics {
        public:
            Steer(int boxSize, int numEntities);
            ~Steer() = default;

            void updatePhysics() override;
            void setSteeringMaxForce(float steeringMaxForce) { m_steeringMaxForce = steeringMaxForce; }
            void setmaxVelocity(float maxVelocity) { m_maxVelocity = maxVelocity; }
            float m_maxVelocity;  
            

        private:
            void generateBoids();
            Math::float3 target;
            float time =0.0f;
            float m_steeringMaxForce;
            Math::float3 seekTarget(Math::float3 location, Entity boid);
            void updateBoid(Entity& boid);
    };
}