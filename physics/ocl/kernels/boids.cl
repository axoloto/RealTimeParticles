
__kernel void matvec_mult(__global float4* matrix, __global float4* vector, __global float* result)
{
  int i = get_global_id(0);
  result[i] = dot(matrix[i], vector[0]);
}

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

  float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0;
  float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0;
  float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0;

  float3 randomXYZ = (float3)(x * step(3.0f, dim), y, z);

  pos[i].xyz = clamp(randomXYZ, (float)-250.0, (float)250.0);
  pos[i].w = 1.0;

  vel[i].xyz = clamp(randomXYZ, (float)-10.0, (float)10.0);
  vel[i].w = 1.0;
}

__constant int MAX_VELOCITY = 5;
__constant int EFFECT_RADIUS = 50;
__constant float MAX_STEERING = 0.5;
__constant float ABS_WALL_POS = 250.0;

__kernel void applyBoidsRules(__global __read_only float4* pos, __global __read_only float4* vel, __global __write_only float4* acc)
{
  unsigned int i = get_global_id(0);
  unsigned int nbEntities = get_global_size(0);

  int count = 0;
  float4 entityPos = pos[i];
  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);
  float dist = 0.0f;
  //float effect = 0.0f;
  for (int e = 0; e < nbEntities; ++e)
  {
    dist = fast_distance(entityPos, pos[e]);

    if (dist < BOIDS_EFFECT_RADIUS && i != e)
    {
      averageBoidsPos += pos[e];
      averageBoidsVel += vel[e];
      repulseHeading += (entityPos - pos[e]) / (dist * dist);
      ++count;
    }

    /*
    effect = (1.0f - step(EFFECT_RADIUS, dist));
    averageBoidsPos += effect * pos[e];
    averageBoidsVel += effect * vel[e];
    repulseHeading += effect * (entityPos - pos[e]) / (dist * dist);
    count += effect * count;
    */
  }

  // cohesion
  averageBoidsPos /= count;
  averageBoidsPos -= entityPos;
  averageBoidsPos = normalize(averageBoidsPos) * MAX_VELOCITY - vel[i];

  // alignment
  averageBoidsVel = normalize(averageBoidsVel) * MAX_VELOCITY - vel[i];

  // separation
  repulseHeading = normalize(repulseHeading) * MAX_VELOCITY - vel[i];

  float4 target = -pos[i];
  acc[i] = clamp(averageBoidsPos, 0.0, normalize(averageBoidsPos) * MAX_STEERING)
      + clamp(averageBoidsVel, 0.0, normalize(averageBoidsVel) * MAX_STEERING)
      + clamp(repulseHeading, 0.0, normalize(repulseHeading) * MAX_STEERING)
      + clamp(target, 0.0, normalize(target) * MAX_STEERING);
}

__kernel void updatePosVerts(__global float4* pos, __global float4* vel, __global __read_only float4* acc)
{
  unsigned int i = get_global_id(0);

  vel[i] += acc[i];
  vel[i] = clamp(vel[i], 0.0, normalize(vel[i]) * MAX_VELOCITY);

  float4 currPos = pos[i] + vel[i];
  float4 clampedCurrPos = clamp(currPos, -ABS_WALL_POS, ABS_WALL_POS);
  if (!all(isequal(clampedCurrPos.xyz, currPos.xyz)))
  {
    vel[i] *= -1;
  }
  pos[i] = clampedCurrPos;
}