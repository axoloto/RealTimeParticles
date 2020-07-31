
#include "BoidsGenerator.hpp"

Core::BoidsGenerator::BoidsGenerator(int halfBoxSize) : m_halfBoxSize(halfBoxSize)
{
    generateBoids();
}

void Core::BoidsGenerator::generateBoids()
{
    for(auto& entity : m_entities)
    {
        float rx = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float ry = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float rz = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

        float x = m_halfBoxSize * (2 * rx - 1.f);
        float y = m_halfBoxSize * (2 * ry - 1.f);
        float z = m_halfBoxSize * (2 * rz - 1.f);

        entity.xyz = {x, y, z};
        entity.rgb = {rx, ry, rz};
    }
}

void* Core::BoidsGenerator::getVerticesBufferStart()
{
    return &m_entities[0];
}

size_t Core::BoidsGenerator::getVerticesBufferSize()
{
    return sizeof(m_entities[0]) * m_entities.size();
}

void Core::BoidsGenerator::updateBoids()
{
    // Where you need to define your physics function with the three Boids rules;
    generateBoids();
}