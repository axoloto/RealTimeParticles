
#include "Boids.hpp"

Core::Boids::Boids(int boxSize, int numEntities) : Physics(boxSize, numEntities)
{
    generateBoids();
}

void Core::Boids::generateBoids()
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

void Core::Boids::updatePhysics()
{

    time += 0.01f;
    target = {sin(time + 4) * m_boxSize / 2, sin(time / 3 + 2) * m_boxSize / 2, sin(time / 4) * m_boxSize / 2};

    for (int i = 0; i < m_numEntities; ++i)
    {
        calculateBoidsForces(m_entities[i], 1.0f, 1.0f, 1.0f);
        //calculateWallForces(m_entities[i], 0.1f);
    }

    for (int i = 0; i < m_numEntities; ++i)
    {
        //m_entities[i].axyz += seekTarget(target, m_entities[i]);

        updateBoid(m_entities[i]);
        cyclingWall(m_entities[i]);
    }
}

Math::float3 Core::Boids::seekTarget(Math::float3 target_location, Entity boid)
{
    Math::float3 steering_force = {0.0f, 0.0f, 0.0f};

    steering_force = normalize(target_location - boid.xyz) * m_maxVelocity - boid.vxyz;

    if (Math::length(steering_force) > m_steeringMaxForce)
    {
        steering_force = normalize(steering_force) * m_steeringMaxForce;
    }
    return steering_force;
}

void Core::Boids::updateBoid(Entity &boid)
{
    boid.vxyz += boid.axyz;
    if (Math::length(boid.vxyz) > m_maxVelocity)
    {
        boid.vxyz = normalize(boid.vxyz) * m_maxVelocity;
    }
    boid.xyz += boid.vxyz;
    boid.axyz = {0.0f, 0.0f, 0.0f};
}

void Core::Boids::calculateBoidsForces(Entity &boid, float scale_separation, float scale_alignment, float scale_cohesion)
{

    Math::float3 separation;
    Math::float3 alignment;
    Math::float3 cohesion;

    float sep_radius = 0.05f * m_boxSize;
    float ali_radius = 0.2f * m_boxSize;
    float coh_radius = 0.25f * m_boxSize;

    separation = {0.0, 0.0, 0.0};
    alignment = {0.0, 0.0, 0.0};
    cohesion = {0.0, 0.0, 0.0};
    float n_coh = 0;
    bool ali = false;
    bool sep = false;

    for (int j = 0; j < m_numEntities; ++j)
    {
        float dist = length(boid.xyz - m_entities[j].xyz);
        if (dist < sep_radius && dist != 0.0f)
        {
            separation += normalize(boid.xyz - m_entities[j].xyz);
            sep = true;
        }
        if (dist < ali_radius && dist != 0.0f)
        {
            alignment += (m_entities[j].vxyz);
            ali = true;
        }

        if (dist < coh_radius && dist != 0.0f)
        {
            n_coh++;
            cohesion += m_entities[j].xyz;
        }
    }

    if (sep){
    separation = normalize(separation) * m_maxVelocity - boid.vxyz;
    if (length(separation) > m_steeringMaxForce)
    {
        separation = normalize(separation) * m_steeringMaxForce;
    }
    }

    if (ali)
    {
        alignment = normalize(alignment) * m_maxVelocity - boid.vxyz;
        if (length(alignment) > m_steeringMaxForce)
        {
            alignment = normalize(alignment) * m_steeringMaxForce;
        }
    }

    if (n_coh > 0)
    {
        cohesion = cohesion / n_coh;
        cohesion = seekTarget(cohesion, boid);
    }
    boid.axyz += separation * scale_separation + alignment * scale_alignment + cohesion * scale_cohesion;
}

void Core::Boids::calculateWallForces(Entity &boid, float ratio)
{
    Math::float3 wall_force = {0, 0, 0};

    float distxp = m_boxSize / 2 - boid.xyz[0];
    float distxn = -m_boxSize / 2 - boid.xyz[0];
    float distyp = m_boxSize / 2 - boid.xyz[1];
    float distyn = -m_boxSize / 2 - boid.xyz[1];
    float distzp = m_boxSize / 2 - boid.xyz[2];
    float distzn = -m_boxSize / 2 - boid.xyz[2];
    float mindist = ratio * m_boxSize / 2;

    if (distxp < mindist)
    {
        wall_force = {-1, 0, 0};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
    else if (-distxn < mindist)
    {
        wall_force = {1, 0, 0};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
    if (distyp < mindist)
    {
        wall_force = {0, -1, 0};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
    else if (-distyn < mindist)
    {
        wall_force = {0, 1, 0};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
    if (distzp < mindist)
    {
        wall_force = {0, 0, -1};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
    else if (-distzn < mindist)
    {
        wall_force = {0, 0, 1};
        wall_force = wall_force * m_maxVelocity  - boid.vxyz;
        boid.axyz += wall_force;
    }
}

void Core::Boids::cyclingWall(Entity &boid)
{
    if(boid.xyz[0]>m_boxSize/2){
        boid.xyz[0]=-boid.xyz[0];
    }
    else if(boid.xyz[0]<-m_boxSize/2){
        boid.xyz[0]=-boid.xyz[0];
    }
        if(boid.xyz[1]>m_boxSize/2){
        boid.xyz[1]=-boid.xyz[0];
    }
    else if(boid.xyz[1]<-m_boxSize/2){
        boid.xyz[1]=-boid.xyz[1];
    }
        if(boid.xyz[2]>m_boxSize/2){
        boid.xyz[2]=-boid.xyz[2];
    }
    else if(boid.xyz[2]<-m_boxSize/2){
        boid.xyz[2]=-boid.xyz[2];
    }
}
