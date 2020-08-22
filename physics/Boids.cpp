
#include "Boids.hpp"

Core::Boids::Boids(int boxSize, int numEntities) : m_maxSteering(4.0f),m_scaleAlignment(1.0f),m_activeSteering(true),m_activeTargets(true),  Physics(boxSize, numEntities)
{
    int boxHalfSize = m_boxSize / 2;
    m_radiusAlignment = m_boxSize * 0.1f;
    Core::Boids::resetBoids(Dimension::dim2D);
}

void Core::Boids::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    for (int i = 0; i < m_numEntities; ++i)
    {
        if (m_activeSteering && m_activeTargets)
        {
            seekTarget(m_entities[i], {0, 0, 0});
            seekTarget(m_entities[i], {0, m_boxSize / 4.0f, 0});
            seekTarget(m_entities[i], {0, -m_boxSize / 4.0f, 0});
            seekTarget(m_entities[i], {0, 0, -m_boxSize / 4.0f});
            seekTarget(m_entities[i], {0, 0, m_boxSize / 4.0f});
        }
        if(m_activeSteering && m_activeAlignment){
            alignment(m_entities[i]);
        }
        updateParticle(m_entities[i]);
        if (m_activateBouncingWall)
        {
            bouncingWall(m_entities[i]);
        }
        if (m_activateCyclicWall)
        {
            cyclicWall(m_entities[i]);
        }
    }
}

Math::float3 Core::Boids::steerForceCalculation(Entity boid, Math::float3 desired_velocity)
{
    Math::float3 steer_force = desired_velocity - boid.vxyz;
    if (length(steer_force) > m_maxSteering)
    {
        steer_force = normalize(steer_force) * m_maxSteering;
    }
    return steer_force;
}

void Core::Boids::seekTarget(Entity &boid, Math::float3 target_loc)
{
    Math::float3 desired_velocity = target_loc - boid.xyz;
    if (length(desired_velocity) > m_maxVelocity)
    {
        desired_velocity = normalize(desired_velocity) * m_maxVelocity;
    }
    boid.axyz += steerForceCalculation(boid, desired_velocity);
}

void Core::Boids::resetBoids(Dimension dim)
{
    float mult = (dim == Dimension::dim3D) ? 1.0f : 0.0f;
    for (int i = 0; i < NUM_MAX_ENTITIES; ++i)
    {
        int boxHalfSize = m_boxSize / 2;

        float rx = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float ry = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float rz = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        float x = boxHalfSize * (2 * rx - 1.f) * mult;
        float y = boxHalfSize * (2 * ry - 1.f);
        float z = boxHalfSize * (2 * rz - 1.f);

        float vx = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * mult;
        float vy = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        float vz = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
        m_entities[i].vxyz = {vx, vy, vz};
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxVelocity;
        m_entities[i].axyz = {0.0f, 0.0f, 0.0f};
    }
}

void Core::Boids::alignment(Entity &boid)
{
    int count = 0;
    Math::float3 averageHeading = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < m_numEntities; ++i)
    {
        float dist = Math::length(boid.xyz - m_entities[i].xyz);
        if (dist < m_radiusAlignment && dist != 0)
        {
            count++;
            averageHeading += m_entities[i].vxyz;
        }
    }
    averageHeading /= float(count);
    boid.axyz += steerForceCalculation(boid, averageHeading);
}