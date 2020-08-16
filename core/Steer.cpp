
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

        float vx = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5);
        float vy = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5);
        float vz = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5);

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
        m_entities[i].vxyz = {vx, vy, vz};
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * 10;
        m_entities[i].axyz = {0, 0, 0};
        target = {0, 0, 0};
    }
}

void Core::Steer::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    //generateBoids();
    Math::float3 steeringforce;
    time+=0.01;
    target={sin(time+4)*m_boxSize/2,sin(time/3+2)*m_boxSize/2,sin(time/4)*m_boxSize/2} ;

    for (int i = 0; i < m_numEntities; ++i)
    {
        m_entities[i].axyz = {0, 0, 0};
        steeringforce = normalize(target - m_entities[i].xyz) * m_maxVelocity - m_entities[i].vxyz;

        if (Math::length(steeringforce) > m_steeringMaxForce)
        {
            steeringforce = normalize(steeringforce) * m_steeringMaxForce;
        }

        m_entities[i].axyz += steeringforce;

        m_entities[i].vxyz += m_entities[i].axyz;
          if (Math::length(m_entities[i].vxyz) > m_maxVelocity)
        {
            m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxVelocity;
        }
        m_entities[i].xyz += m_entities[i].vxyz;
    }
}