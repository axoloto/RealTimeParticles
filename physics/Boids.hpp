#pragma once

#include <array>
#include "Physics.hpp"

namespace Core
{

    class Boids : public Physics
    {
    public:
        Boids(int boxSize, int numEntities);
        ~Boids() = default;

        void updatePhysics() override;

        void setMaxSteering(float maxSteering) { m_maxSteering = maxSteering; }
        float getmaxSteering() { return m_maxSteering; }

        void setRadiusAlignment(float alignment) { m_radiusAlignment = alignment; }
        float getRadiusAlignment() { return m_radiusAlignment; }

        void setRadiusCohesion(float cohesion) { m_radiusCohesion = cohesion; }
        float getRadiusCohesion() { return m_radiusCohesion; }

        void setRadiusSeparation(float separation) { m_radiusSeparation = separation; }
        float getRadiusSeparation() { return m_radiusSeparation; }

        void setScaleAlignment(float alignment) { m_scaleAlignment = alignment; }
        float getScaleAlignment() { return m_scaleAlignment; }

        void setScaleCohesion(float cohesion) { m_scaleCohesion = cohesion; }
        float getScaleCohesion() { return m_scaleCohesion; }

        void setScaleSeparation(float separation) { m_scaleSeparation = separation; }
        float getScaleSeparation() { return m_scaleSeparation; }

        void setSteering(bool steering) { m_activeSteering = steering; }
        bool getSteering() { return m_activeSteering; }

        void setActivateTargets(bool targets) { m_activeTargets = targets; }
        bool getActivateTargets() { return m_activeTargets; }

        void setActivateAlignment(bool alignment) { m_activeAlignment = alignment; }
        bool getActivateAlignment() { return m_activeAlignment; }

        void setActivateCohesion(bool cohesion) { m_activeCohesion = cohesion; }
        bool getActivateCohesion() { return m_activeCohesion; }

        void setActivateSeparation(bool separation) { m_activeSeparation = separation; }
        bool getActivateSeparation() { return m_activeSeparation; }

    private:
        Math::float3 steerForceCalculation(Entity boid, Math::float3 desired_velocity);
        void seekTarget(Entity &boid, Math::float3 target_loc,float scale);
        void alignment(Entity &boid);
        void cohesion(Entity &boid);
        void separation(Entity &boid);
        void repulseWall(Entity &boid); // WIP
        float m_maxSteering;
        float m_radiusAlignment;
        float m_scaleAlignment;
        float m_radiusCohesion;
        float m_scaleCohesion;
        float m_radiusSeparation;
        float m_scaleSeparation;
        bool m_activeSteering;
        bool m_activeTargets;
        bool m_activeAlignment;
        bool m_activeCohesion;
        bool m_activeSeparation;
    };
} // namespace Core
