
#include "Boids.hpp"


Core::Boids::Boids(int boxSize, int numEntities) : Physics(boxSize, numEntities)
{
    generateBoids();
}

void Core::Boids::generateBoids()
{
    int boxHalfSize = m_boxSize / 2;
    m_maxVelocity=0.5f;
    m_maxSteering=4.0f;
    m_bouncingwall=true;
    m_steering=true;
    m_forcedmaxspeed=true;
    m_pause=true;

    for(int i = 0; i < NUM_MAX_ENTITIES; ++i) 
    {
        float rx = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float ry = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        float rz = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

        float x = 0.0f;//boxHalfSize * (2 * rx - 1.f);
        float y = boxHalfSize * (2 * ry - 1.f);
        float z = boxHalfSize * (2 * rz - 1.f);

        float vx = 0.0f;//(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        float vy = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        float vz = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
        m_entities[i].vxyz = {vx, vy, vz};
        m_entities[i].vxyz = normalize(m_entities[i].vxyz) * m_maxVelocity;
        m_entities[i].axyz = {0.0f, 0.0f, 0.0f};

        m_entities[i].xyz = {x, y, z};
        m_entities[i].rgb = {rx, ry, rz};
    }
}

void Core::Boids::updatePhysics()
{
    // Where you need to define your physics function with the three Boids rules;
    for(int i = 0; i < m_numEntities; ++i) {
    if(m_steering){seekTarget(m_entities[i],{0,0,0});}
    updateBoid(m_entities[i]);
    if (m_bouncingwall){bouncingWall(m_entities[i]);}
    }
}



void Core::Boids::updateBoid(Entity &boid)
{
    if(!m_pause){
    boid.vxyz += boid.axyz;
    
    if (Math::length(boid.vxyz) > m_maxVelocity || m_forcedmaxspeed) {
        boid.vxyz = normalize(boid.vxyz) * m_maxVelocity;
    }
    boid.xyz += boid.vxyz;
    boid.axyz = {0.0f, 0.0f, 0.0f};
    }
}

void Core::Boids::bouncingWall(Entity &boid){

    if (abs(boid.xyz[0])>m_boxSize/2){boid.vxyz[0]=-boid.vxyz[0];}
    if (abs(boid.xyz[1])>m_boxSize/2){boid.vxyz[1]=-boid.vxyz[1];}
    if (abs(boid.xyz[2])>m_boxSize/2){boid.vxyz[2]=-boid.vxyz[2];}

}

Math::float3 Core::Boids::steerForceCalculation(Entity boid, Math::float3 desired_velocity){
    Math::float3 steer_force=desired_velocity-boid.vxyz;
    if (length(steer_force)>m_maxSteering){steer_force=normalize(steer_force)*m_maxSteering;}
    return steer_force;
}

void Core::Boids::seekTarget(Entity &boid, Math::float3 target_loc){
    Math::float3 desired_velocity=target_loc-boid.xyz;
    if (length(desired_velocity)>m_maxVelocity){desired_velocity=normalize(desired_velocity)*m_maxVelocity;}
    boid.axyz += steerForceCalculation(boid,desired_velocity);
}


