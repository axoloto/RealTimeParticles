#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Boids : public Physics {
        public:
            Boids(int boxSize, int numEntities);
            ~Boids() = default;

            void updatePhysics() override;
            void setmaxVelocity(float maxVelocity) { m_maxVelocity = maxVelocity; }
            float getmaxVelocity() { return m_maxVelocity; }
            void setmaxSteering(float maxSteering) { m_maxSteering = maxSteering; }
            float getmaxSteering() { return m_maxSteering; }           
            void setbouncingWall(bool bouncingwall) { m_bouncingwall = bouncingwall; }
            float getbouncingWall() { return m_bouncingwall; }
            void setSteering(bool steering) { m_steering = steering; }
            float getSteering() { return m_steering; }
            void setForcedMaxSpeed(bool forcedmax) { m_forcedmaxspeed = forcedmax; }
            float getForcedMaxspeed() { return m_forcedmaxspeed; }
            void setPause(bool pause) { m_pause = pause; }
            float getpause() { return m_pause; }
            void resetBoids2D();
            void resetBoids3D();

        private:
            void generateBoids();
            void updateBoid(Entity& boid);
            void bouncingWall(Entity& boid);
            Math::float3 steerForceCalculation(Entity boid,Math::float3 desired_velocity);
            void seekTarget(Entity& boid,Math::float3 target_loc);

            float m_maxVelocity;
            float m_maxSteering;
            bool m_bouncingwall;
            bool m_steering;
            bool m_forcedmaxspeed;
            bool m_pause;
    };
}


