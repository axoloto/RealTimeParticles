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
            void setSteering(bool steering) { m_activeSteering = steering; }
            float getSteering() { return m_activeSteering; }
            void resetBoids(Dimension dim);
            void setActivateTargets(bool targets) { m_activeTargets = targets; }
            float getActivateTargets() { return m_activeTargets; }
            void setActivateAlignment(bool alignment) { m_activeAlignment = alignment; }
            float getActivateAlignment() { return m_activeAlignment; }

        private:
            

            Math::float3 steerForceCalculation(Entity boid,Math::float3 desired_velocity);
            void seekTarget(Entity& boid,Math::float3 target_loc);
            void alignment(Entity& boid);
            void repulseWall(Entity& boid); // WIP
            float m_maxSteering;
            float m_radiusAlignment;
            float m_scaleAlignment;
            bool m_activeSteering;
            bool m_activeTargets;
            bool m_activeAlignment;
    };
}


