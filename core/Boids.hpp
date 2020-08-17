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
            void calculateBoidsForces(Entity& boid,float scale_separation, float scale_alignment, float scale_cohesion);
            Math::float3 seekTarget(Math::float3 location, Entity boid);
            void updateBoid(Entity& boid);
            void calculateWallForces(Entity& boid,float ratio);
            void cyclingWall(Entity& boid);
    };
}


 