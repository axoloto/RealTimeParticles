
#include "Boids.hpp"

Core::Boids::Boids(int boxSize, int numEntities) : Physics(boxSize, numEntities)
{
    int boxHalfSize = m_boxSize / 2;
    m_maxVelocity = 0.5f;
    m_maxSteering = 4.0f;
    m_radiusAlignment = m_boxSize*0.1f;
    m_scaleAlignment = 1.0f;
    m_bouncingWall = true;
    m_steering = true;
    m_forcedmaxspeed = true;
    m_pause = true;
    Core::Boids::resetBoids(2);
}

void Core::Boids::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    for (int i = 0; i < m_numEntities; ++i)
    {
        if (m_steering)
        {
            seekTarget(m_entities[i], {0, 0, 0});
            seekTarget(m_entities[i], {0, m_boxSize/4.0f, 0});
            seekTarget(m_entities[i], {0, -m_boxSize/4.0f, 0});
            seekTarget(m_entities[i], {0, 0, -m_boxSize/4.0f});
            seekTarget(m_entities[i], {0, 0,m_boxSize/4.0f});
        }
        alignment(m_entities[i]);
        updateBoid(m_entities[i]);
        if (m_bouncingWall)
        {
            bouncingWall(m_entities[i]);
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

void Core::Boids::resetBoids(int dim)
{
    if (dim == 2 || dim == 3)
    {
        for (int i = 0; i < NUM_MAX_ENTITIES; ++i)
        {
            int boxHalfSize = m_boxSize / 2;

            float rx = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float ry = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float rz = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            float x = boxHalfSize * (2 * rx - 1.f) * (dim - 2);
            ;
            float y = boxHalfSize * (2 * ry - 1.f);
            float z = boxHalfSize * (2 * rz - 1.f);

            float vx = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * (dim - 2);
            float vy = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
            float vz = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);

            m_entities[i].xyz = {x, y, z};
            m_entities[i].rgb = {rx, ry, rz};
            m_entities[i].vxyz = {vx, vy, vz};
            m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxVelocity;
            m_entities[i].axyz = {0.0f, 0.0f, 0.0f};
        }
    }
}

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
        m_entities[i].vxyz = {vx, vy, vz};
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxVelocity;
        m_entities[i].axyz = {0.0f, 0.0f, 0.0f};
    }
}