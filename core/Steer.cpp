
#include "Steer.hpp"

Core::Steer::Steer(int boxSize, int numEntities) : Physics(boxSize, numEntities)
{
    generateBoids();
}

void Core::Steer::generateBoids()
{
    int boxHalfSize = m_boxSize / 2;

    for (int i = 0; i < NUM_MAX_ENTITIES; ++i)
    {
        float rx = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float ry = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float rz = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        float x = boxHalfSize * (2 * rx - 1.f);
        float y = boxHalfSize * (2 * ry - 1.f);
        float z = boxHalfSize * (2 * rz - 1.f);

        float vx = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        float vy = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        float vz = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
        m_entities[i].vxyz = {vx, vy, vz};
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * 10;
        m_entities[i].axyz = {0.0f, 0.0f, 0.0f};
        target = {0.0f, 0.0f, 0.0f};
    }
}

void Core::Steer::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    //generateBoids();

    time += 0.01f;
    target = {sin(time + 4) * m_boxSize / 2, sin(time / 3 + 2) * m_boxSize / 2, sin(time / 4) * m_boxSize / 2};

    for (int i = 0; i < m_numEntities; ++i)
    {
        m_entities[i].axyz += seekTarget(target, m_entities[i]);
        updateBoid(m_entities[i]);
    }
}

Math::float3 Core::Steer::seekTarget(Math::float3 target_location, Entity boid)
{
    Math::float3 steering_force = {0.0f, 0.0f, 0.0f};

    steering_force = normalize(target - boid.xyz) * m_maxVelocity - boid.vxyz;

    if (Math::length(steering_force) > m_steeringMaxForce)
    {
        steering_force = normalize(steering_force) * m_steeringMaxForce;
    }
    return steering_force;
}

void Core::Steer::updateBoid(Entity& boid){
        boid.vxyz += boid.axyz;
        if (Math::length(boid.vxyz) > m_maxVelocity)
        {
            boid.vxyz = normalize(boid.vxyz) * m_maxVelocity;
        }
        boid.xyz += boid.vxyz;
        boid.axyz = {0.0f, 0.0f, 0.0f};
}