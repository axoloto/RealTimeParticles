
#include "Physics.hpp"

Core::Physics::Physics(int boxSize, int numEntities) : m_boxSize(boxSize), m_numEntities(numEntities){}

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
    if(m_numEntities > NUM_MAX_ENTITIES) return;

    for(int i = 0; i < m_numEntities; ++i) 
    {
        m_coordsBuffer[i] = { m_entities[i].xyz[0], m_entities[i].xyz[1], m_entities[i].xyz[2] } ;
        m_colorsBuffer[i] = { m_entities[i].rgb[0], m_entities[i].rgb[1], m_entities[i].rgb[2] };
    }
}