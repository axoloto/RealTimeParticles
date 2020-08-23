
#include "Boids.hpp"

Core::Boids::Boids(int boxSize, int numEntities) : m_maxSteering(0.5f),m_scaleAlignment(1.0f),m_scaleCohesion(1.0f),m_scaleSeparation(1.0f),m_activeSteering(true),m_activeTargets(true),
m_activeAlignment(true), m_activeSeparation(true),m_activeCohesion(true),  Physics(boxSize, numEntities)
{
    int boxHalfSize = m_boxSize / 2;
    m_radiusAlignment = m_boxSize * 0.1f;
    m_radiusCohesion = m_boxSize * 0.1f;
    m_radiusSeparation = m_boxSize * 0.1f;
    Core::Boids::resetParticle(Dimension::dim2D);
}

void Core::Boids::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    for (int i = 0; i < m_numEntities; ++i)
    {
        if (m_activeSteering && m_activeTargets)
        {
            seekTarget(m_entities[i], {0, 0, 0},1);
            seekTarget(m_entities[i], {0, m_boxSize / 4.0f, 0},1);
            seekTarget(m_entities[i], {0, -m_boxSize / 4.0f, 0},1);
            seekTarget(m_entities[i], {0, 0, -m_boxSize / 4.0f},1);
            seekTarget(m_entities[i], {0, 0, m_boxSize / 4.0f},1);
        }
        if(m_activeSteering && m_activeAlignment){
            alignment(m_entities[i]);
        }
        if(m_activeSteering && m_activeCohesion){
            cohesion(m_entities[i]);
        }
        if(m_activeSteering && m_activeSeparation){
            separation(m_entities[i]);
        }
    }
    for (int i = 0; i < m_numEntities; ++i)
    {
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

void Core::Boids::seekTarget(Entity &boid, Math::float3 target_loc , float scale)
{
    Math::float3 desired_velocity = target_loc - boid.xyz;
    if (length(desired_velocity) > m_maxVelocity)
    {
        desired_velocity = normalize(desired_velocity) * m_maxVelocity;
    }
    boid.axyz += steerForceCalculation(boid, desired_velocity)*scale;
}



void Core::Boids::alignment(Entity &boid)
{
    int count =0;
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
    if (count>0){
    averageHeading=normalize(averageHeading)*m_maxVelocity;
    boid.axyz += steerForceCalculation(boid, averageHeading)*m_scaleAlignment;
    }
}

void Core::Boids::cohesion(Entity &boid)
{
    int count=0;
    Math::float3 averagePosition = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < m_numEntities; ++i)
    {
        float dist = Math::length(boid.xyz - m_entities[i].xyz);
        if (dist < m_radiusCohesion && dist != 0)
        {   count++;
            averagePosition += m_entities[i].xyz;
        }
    }
    if (count>0){
    averagePosition /=float(count);
    averagePosition -=boid.xyz;
    averagePosition =normalize(averagePosition)*m_maxVelocity;
    boid.axyz += steerForceCalculation(boid, averagePosition)*m_scaleCohesion;
    }
}

void Core::Boids::separation(Entity &boid)
{
    int count = 0;
    Math::float3 repulseHeading = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < m_numEntities; ++i)
    {
        float dist = Math::length(boid.xyz - m_entities[i].xyz);
        if (dist < m_radiusSeparation && dist != 0)
        {
            count++;
            repulseHeading += (boid.xyz - m_entities[i].xyz)/(dist*dist);
        }
    }
    if (count>0){
    repulseHeading = normalize(repulseHeading)*m_maxVelocity;
    }
    boid.axyz += steerForceCalculation(boid, repulseHeading)*m_scaleSeparation;
}