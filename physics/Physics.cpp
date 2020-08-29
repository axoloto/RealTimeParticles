
#include "Physics.hpp"

Core::Physics::Physics(int boxSize, int numEntities, Dimension dimension) : m_boxSize(boxSize), m_numEntities(numEntities), m_maxSpeed(4.0f), m_dimension(dimension),
                                                                            m_activateBouncingWall(false), m_activateCyclicWall(true), m_forceMaxSpeed(true), m_pause(true) {}

void* Core::Physics::getCoordsBufferStart()
{
    return &m_coordsBuffer[0];
}

void* Core::Physics::getColorsBufferStart()
{
    return &m_colorsBuffer[0];
}

void Core::Physics::update()
{
    updatePhysics();
    updateBuffers();
}

void Core::Physics::updateBuffers()
{
    if (m_numEntities > NUM_MAX_ENTITIES) return;

    for (int i = 0; i < m_numEntities; ++i)
    {
        m_coordsBuffer[i] = {m_entities[i].xyz[0], m_entities[i].xyz[1], m_entities[i].xyz[2]};
        m_colorsBuffer[i] = {m_entities[i].rgb[0], m_entities[i].rgb[1], m_entities[i].rgb[2]};
    }
}

void Core::Physics::updateParticle(Entity &particle)
{
    if (!m_pause)
    {
        particle.vxyz += particle.axyz;

        if (Math::length(particle.vxyz) > m_maxSpeed || m_forceMaxSpeed)
        {
            particle.vxyz = normalize(particle.vxyz) * m_maxSpeed;
        }
        particle.xyz += particle.vxyz;
        particle.axyz = {0.0f, 0.0f, 0.0f};
    }
}

void Core::Physics::bouncingWall(Entity &particle)
{
    if (abs(particle.xyz[0]) > m_boxSize / 2)
    {
        particle.vxyz[0] *= -1;
    }
    if (abs(particle.xyz[1]) > m_boxSize / 2)
    {
        particle.vxyz[1] *= -1;
    }
    if (abs(particle.xyz[2]) > m_boxSize / 2)
    {
        particle.vxyz[2] *= -1;
    }
}

void Core::Physics::cyclicWall(Entity &particle)
{
    float wall = m_boxSize / 2.0f;

    for (int i = 0; i < 3; i++)
    {
        float overshoot = abs(particle.xyz[i]) - wall;
        if (overshoot > 0)
        {
            if (particle.xyz[i] > 0)
            {
                particle.xyz[i] = -particle.xyz[i] + overshoot;
            }
            else
            {
                particle.xyz[i] = -particle.xyz[i] - overshoot;
            }
        }
    }
}

void Core::Physics::resetParticles()
{
    float mult = (m_dimension == Dimension::dim3D) ? 1.0f : 0.0f;
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
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxSpeed;
        m_entities[i].axyz = {0.0f, 0.0f, 0.0f};
    }
}