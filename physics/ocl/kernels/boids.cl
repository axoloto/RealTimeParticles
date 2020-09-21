
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

__kernel void randPosVerts(__global float4* pos)
{
  unsigned int i = get_global_id(0);

  unsigned int randomIntX = parallelRNG(i);
  unsigned int randomIntY = parallelRNG(i + 1);
  unsigned int randomIntZ = parallelRNG(i + 2);

  float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0;
  float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0;
  float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0;

  float3 randomCoords = (float3)(x, y, z);

  pos[i].xyz = clamp(randomCoords, (float)-250.0, (float)250.0);
  pos[i].w = 1.0;
}

__constant int MAX_VELOCITY = 5;
__constant int EFFECT_RADIUS = 50;
__constant float MAX_STEERING = 0.5;
__constant float BOX_SIZE = 500.0;

__kernel void applyBoidsRules(__global float4* pos, __global float4* vel, __global float4* acc)
{
  unsigned int i = get_global_id(0);
  unsigned int nbEntities = get_global_size(0);

  int count = 0;
  float3 entityPos = pos[i].xyz;
  float3 averageBoidsPos = (float3)(0.0, 0.0, 0.0);
  float3 averageBoidsVel = (float3)(0.0, 0.0, 0.0);
  float3 repulseHeading = (float3)(0.0, 0.0, 0.0);
  for (int e = 0; e < nbEntities; ++e)
  {
    if (e == i)
      continue;

    float dist = fast_distance(entityPos, pos[e].xyz);
    if (dist < EFFECT_RADIUS)
    {
      averageBoidsPos += pos[e].xyz;
      averageBoidsVel += vel[e].xyz;
      repulseHeading += (entityPos - pos[e].xyz) / (dist * dist);
      ++count;
    }
  }

  // cohesion
  averageBoidsPos /= count;
  averageBoidsPos -= entityPos;
  averageBoidsPos = normalize(averageBoidsPos) * MAX_VELOCITY - vel[i].xyz;

  // alignment
  averageBoidsVel = normalize(averageBoidsVel) * MAX_VELOCITY - vel[i].xyz;

  // separation
  repulseHeading = normalize(repulseHeading) * MAX_VELOCITY - vel[i].xyz;

  acc[i].xyz = clamp(averageBoidsPos, 0.0, normalize(averageBoidsPos) * MAX_STEERING)
      + clamp(averageBoidsVel, 0.0, normalize(averageBoidsVel) * MAX_STEERING)
      + clamp(repulseHeading, 0.0, normalize(repulseHeading) * MAX_STEERING);
}

__kernel void updatePosVerts(__global float4* pos, __global float4* vel, __global float4* acc)
{
  unsigned int i = get_global_id(0);

  vel[i] += acc[i];
  vel[i] = clamp(vel[i], 0.0, normalize(vel[i]) * MAX_VELOCITY);

  if (isgreaterequal(fabs(pos[i].x), (float)(BOX_SIZE * 0.5)) != 0)
  {
    vel[i].x *= -1;
  }

  if (isgreaterequal(fabs(pos[i].y), (float)(BOX_SIZE * 0.5)) != 0)
  {
    vel[i].y *= -1;
  }

  if (isgreaterequal(fabs(pos[i].z), (float)(BOX_SIZE * 0.5)) != 0)
  {
    vel[i].z *= -1;
  }

  pos[i] += vel[i];
}