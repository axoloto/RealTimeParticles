
#include "BoidsGenerator.hpp"

Core::BoidsGenerator::BoidsGenerator(int boxSize, int numEntities) : m_boxSize(boxSize), m_numEntities(numEntities)
{
    generateBoids();
}

void Core::BoidsGenerator::generateBoids()
{
    int boxHalfSize = m_boxSize / 2;

    for(int i = 0; i < m_numEntities; ++i) 
    {
        float rx = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float ry = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float rz = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

        float x = boxHalfSize * (2 * rx - 1.f);
        float y = boxHalfSize * (2 * ry - 1.f);
        float z = boxHalfSize * (2 * rz - 1.f);

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
    }
}

void* Core::BoidsGenerator::getVerticesBufferStart()
{
    return &m_entities[0];
}

size_t Core::BoidsGenerator::getVerticesBufferSize()
{
    // Always give the full buffer for now
    return sizeof(m_entities[0]) * m_entities.size();
}

void Core::BoidsGenerator::updateBoids()
{
    // Where you need to define your physics function with the three Boids rules;
    generateBoids();
}