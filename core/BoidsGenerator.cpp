
#include "BoidsGenerator.hpp"

Core::BoidsGenerator::BoidsGenerator(int boxSize, int numEntities) : m_boxSize(boxSize), m_numEntities(numEntities)
{
    generateBoids();
}

void Core::BoidsGenerator::generateBoids()
{
    int boxHalfSize = m_boxSize / 2;

    for (auto &phy_boids : m_phy_boids)
    {
        float x = boxHalfSize * (2 * randomZeroOne() - 1.f);
        float y = boxHalfSize * (2 * randomZeroOne() - 1.f);
        float z = boxHalfSize * (2 * randomZeroOne() - 1.f);
        //float vlim = (randomZeroOne()-0.5f)*10;
        float vlim = 1.0f;
        float vx = randomZeroOne() - 0.5f;
        float vy = randomZeroOne() - 0.5f;
        float vz = randomZeroOne() - 0.5f;
        float norm = norm3D(vx, vy, vz);
        //norm=1;
        vx = (vx / norm) * vlim;
        vy = (vy / norm) * vlim;
        vz = (vz / norm) * vlim;

        phy_boids.xyz = {x, y, z};
        phy_boids.vxyz = {vx, vy, vz};
    }

    for (auto &entity : m_entities)
    {
        float rx = randomZeroOne();
        float ry = randomZeroOne();
        float rz = randomZeroOne();

        //float x = boxHalfSize* (2 *  randomZeroOne()- 1.f);
        //float y = boxHalfSize * (2 * randomZeroOne() - 1.f);
        //float z = boxHalfSize * (2 * randomZeroOne() - 1.f);
        //m_entities[i].xyz = {x, y, z};
        entity.rgb = {rx, ry, rz};
    }
    UpdateCoord();
}

void *Core::BoidsGenerator::getVerticesBufferStart()
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
    //generateBoids();

    UpdateCoord();
    WallCyclic();
    /*m_entities[i].xyz[0]+=m_entities[i].vxyz[0];
        m_entities[i].xyz[1]+=m_entities[i].vxyz[1];
        m_entities[i].xyz[2]+=m_entities[i].vxyz[2];*/

    //COLLIDE WALL FUNCTION TO BUILD
}

float Core::BoidsGenerator::randomZeroOne()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float Core::BoidsGenerator::norm3D(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

void Core::BoidsGenerator::UpdateCoord()
{
    for (int i = 0; i < m_numEntities; ++i)
    {

        m_phy_boids[i].xyz[0] += m_phy_boids[i].vxyz[0];
        m_phy_boids[i].xyz[1] += m_phy_boids[i].vxyz[1];
        m_phy_boids[i].xyz[2] += m_phy_boids[i].vxyz[2];
        m_entities[i].xyz = m_phy_boids[i].xyz;
    }
}

void Core::BoidsGenerator::WallCyclic()
{
    for (int i = 0; i < m_numEntities; ++i)
    {
        if (abs(m_phy_boids[i].xyz[0]) > m_boxSize / 2)
        {
            m_phy_boids[i].xyz[0] = -m_phy_boids[i].xyz[0];
        }
        if (abs(m_phy_boids[i].xyz[1]) > m_boxSize / 2)
        {
            m_phy_boids[i].xyz[1] = -m_phy_boids[i].xyz[1];
        }
        if (abs(m_phy_boids[i].xyz[2]) > m_boxSize / 2)
        {
            m_phy_boids[i].xyz[2] = -m_phy_boids[i].xyz[2];
        }
    }
}