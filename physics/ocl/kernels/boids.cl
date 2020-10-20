unsigned int parallelRNG(unsigned int i)
{
  unsigned int value = i;

  value = (value ^ 61) ^ (value >> 16);
  value *= 9;
  value ^= value << 4;
  value *= 0x27d4eb2d;
  value ^= value >> 15;

  return value;
}

__kernel void colorVerts(__global float4* color)
{
  int i = get_global_id(0);
  float col = i / (float)get_global_size(0);
  color[i] = (float4)(col, col, col, 1.0);
}

__kernel void randPosVerts(__global float4* pos, __global float4* vel, float dim)
{
  unsigned int i = get_global_id(0);

  unsigned int randomIntX = parallelRNG(i);
  unsigned int randomIntY = parallelRNG(i + 1);
  unsigned int randomIntZ = parallelRNG(i + 2);

  float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0f;
  float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0f;
  float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0f;

  float3 randomXYZ = (float3)(x * step(3.0f, dim), y, z);

  pos[i].xyz = clamp(randomXYZ, -250.0f, 250.0f);
  pos[i].w = 1.0;

  vel[i].xyz = clamp(randomXYZ, -10.0f, 10.0f);
  vel[i].w = 1.0;
}

typedef struct
{
  float velocity;
  float scaleCohesion;
  float scaleAlignment;
  float scaleSeparation;
  int activeTarget;
} boidsParams;

inline float4 steerForce(float4 desiredVel, float4 vel)
{
  float4 steerForce = desiredVel - vel;
  if (length(steerForce) > BOIDS_MAX_STEERING)
  {
    steerForce = normalize(steerForce) * BOIDS_MAX_STEERING;
  }
  return steerForce;
}

__kernel void applyBoidsRules(__global float4* position, __global float4* velocity, __global float4* acc, __global boidsParams* params)
{
  unsigned int i = get_global_id(0);
  unsigned int numEnt = get_global_size(0);

  float4 pos = position[i];
  float4 vel = velocity[i];

  int count = 0;

  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int e = 0; e < numEnt; ++e)
  {
    vec = pos - position[e];
    squaredDist = dot(vec, vec);

    if (squaredDist < BOIDS_EFFECT_RADIUS_SQUARED && i != e)
    {
      averageBoidsPos += position[e];
      averageBoidsVel += velocity[e];
      repulseHeading += vec / squaredDist;
      ++count;
    }
  }

  if (count != 0)
  {
    // cohesion
    averageBoidsPos /= count;
    averageBoidsPos -= pos;
    averageBoidsPos = normalize(averageBoidsPos) * params->velocity;
    // alignment
    averageBoidsVel = normalize(averageBoidsVel) * params->velocity;
    // separation
    repulseHeading = normalize(repulseHeading) * params->velocity;
  }

  float4 target = -pos;

  acc[i] = steerForce(averageBoidsPos, vel) * params->scaleCohesion
      + steerForce(averageBoidsVel, vel) * params->scaleAlignment
      + steerForce(repulseHeading, vel) * params->scaleSeparation
      + clamp(target, 0.0, normalize(target) * BOIDS_MAX_STEERING) * params->activeTarget;
}

__kernel void updateVelVerts(__global float4* vel, __global float4* acc, __global boidsParams* params)
{
  unsigned int i = get_global_id(0);

  vel[i] += acc[i];

  vel[i] = normalize(vel[i]) * params->velocity;
}

__kernel void updatePosVertsWithBouncingWalls(__global float4* pos, __global float4* vel)
{
  unsigned int i = get_global_id(0);

  float4 newPos = pos[i] + vel[i];
  float4 clampedNewPos = clamp(newPos, -ABS_WALL_POS, ABS_WALL_POS);
  if (!all(isequal(clampedNewPos.xyz, newPos.xyz)))
  {
    vel[i] *= -1;
  }
  pos[i] = clampedNewPos;
}

__kernel void updatePosVertsWithCyclicWalls(__global float4* pos, __global float4* vel)
{
  unsigned int i = get_global_id(0);

  float4 newPos = pos[i] + vel[i];
  float4 clampedNewPos = clamp(newPos, -ABS_WALL_POS, ABS_WALL_POS);
  if (!isequal(clampedNewPos.x, newPos.x))
  {
    clampedNewPos.x *= -1;
  }
  if (!isequal(clampedNewPos.y, newPos.y))
  {
    clampedNewPos.y *= -1;
  }
  if (!isequal(clampedNewPos.z, newPos.z))
  {
    clampedNewPos.z *= -1;
  }
  pos[i] = clampedNewPos;
}