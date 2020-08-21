#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Boids : public Physics {
        public:
            Boids(int boxSize, int numEntities);
            ~Boids() = default;

            void updatePhysics() override;
            void setmaxSteering(float maxSteering) { m_maxSteering = maxSteering; }
            float getmaxSteering() { return m_maxSteering; } 
            void setSteering(bool steering) { m_steering = steering; }
            float getSteering() { return m_steering; }
            void resetBoids(int dim);

        private:
            

            Math::float3 steerForceCalculation(Entity boid,Math::float3 desired_velocity);
            void seekTarget(Entity& boid,Math::float3 target_loc);

            float m_maxVelocity;
            float m_maxSteering;
            float m_radiusAlignment;
            float m_scaleAlignment;
            bool m_steering;
    };
}


