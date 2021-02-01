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

inline float4 steerForce(float4 desiredVel, float4 vel)
{
  float4 steerForce = desiredVel - vel;
  if (length(steerForce) > MAX_STEERING)
  {
    steerForce = normalize(steerForce) * MAX_STEERING;
  }
  return steerForce;
}

/*
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

    // Second condition to deal with almost identical points generated by parallelRNG and i == e
    if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
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
      + clamp(target, 0.0, normalize(target) * MAX_STEERING) * params->activeTarget;

  // Dealing with numerical error, forcing 2D
  if (params->dims < 3.0f)
    acc[i].x = 0.0f;
}
*/

__kernel void updateVelVerts(__global float4* vel, __global float4* acc, float velVal)
{
  unsigned int i = get_global_id(0);

  vel[i] += acc[i];

  vel[i] = normalize(vel[i]) * velVal;
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

// For rendering purpose only, checking for each grid cell if there is any particle
// If so, grid cell will be displayed in OpenGl
__kernel void flushGridDetector(__global float8* gridDetector)
{
  unsigned int i = get_global_id(0);
  gridDetector[i] = 0.0f;
}

int3 getCell3DIndex(float4 pos)
{
  int cellSize = 2 * ABS_WALL_POS / GRID_RES;

  // Moving particles in [0 - 2 * ABS_WALL_POS] to have coords matching with cellIndices
  // Adding epsilon to avoid wrong indices if particle exactly on the ABS_WALL_POS
  float3 posXYZ = pos.xyz + ABS_WALL_POS - FLOAT_EPSILON;

  int3 cell3DIndex = convert_int3(posXYZ / cellSize);

  clamp(cell3DIndex, 0, GRID_RES - 1);

  return cell3DIndex;
}

__kernel void fillGridDetector(__global float4* vertPos, __global float8* gridDetector)
{
  unsigned int i = get_global_id(0);

  float4 pos = vertPos[i];

  int3 cellIndex = getCell3DIndex(pos);

  int gridDetectorIndex = cellIndex.x * GRID_RES * GRID_RES
      + cellIndex.y * GRID_RES
      + cellIndex.z;

  if (gridDetectorIndex < GRID_RES * GRID_RES * GRID_RES) // WIP
    gridDetector[gridDetectorIndex] = 1.0f;
}

// To use of Radix Sort accelerator, we need to find the cellID for each boids particle
__kernel void flushCellIDs(__global uint* boidsCellIDs)
{
  unsigned int i = get_global_id(0);
  // For all particles, giving cell ID above any available one
  // the ones not filled later (i.e not processed because index > numEntites displayed)
  // will be sorted at the end and not considered after sorting
  boidsCellIDs[i] = GRID_NUM_CELLS + 1;
}

__kernel void fillCellIDs(__global float4* vertPos, __global uint* boidsCellIDs)
{
  unsigned int i = get_global_id(0);

  float4 pos = vertPos[i];

  int3 cell3DIndex = getCell3DIndex(pos);

  uint cell1DIndex = cell3DIndex.x * GRID_RES * GRID_RES
      + cell3DIndex.y * GRID_RES
      + cell3DIndex.z;

  boidsCellIDs[i] = cell1DIndex;
}

__kernel void flushStartEndCell(__global uint2* startEndCells)
{
  unsigned int i = get_global_id(0);

  // Flushing buffer with 1 as starting index and 0 as ending index
  // This little trick bypass any loop where
  // we loop on the particles found in a cell
  // if this cell has not been found, hence has no particle at all
  //startEndCells[i].xy = (uint)(1, 0);
  startEndCells[i].x = 1;
  startEndCells[i].y = 0;
}

__kernel void fillStartCell(__global uint* boidsCellIDs, __global uint2* startEndCells)
{
  unsigned int i = get_global_id(0);

  uint currentCellID = boidsCellIDs[i];

  if (i > 0)
  {
    uint leftCellID = boidsCellIDs[i - 1];
    if (currentCellID != leftCellID)
    {
      // Found start
      startEndCells[currentCellID].x = i;
    }
  }
}

__kernel void fillEndCell(__global uint* boidsCellIDs, __global uint2* startEndCells)
{
  unsigned int i = get_global_id(0);

  uint currentCellID = boidsCellIDs[i];

  if (i != get_global_size(0))
  {
    uint rightCellID = boidsCellIDs[i + 1];
    if (currentCellID != rightCellID)
    {
      // Found end
      startEndCells[currentCellID].y = i;
    }
  }
}
/*
__kernel void applyBoidsRulesWithGridSmart(
    __global float4* position,
    __global float4* velocity,
    __global float4* acceleration,
    __global uint2* startEndCell,
    __local float4* pos
        float8 params)
{
  unsigned int gid = get_group_id(0); // cell

  uint2 startEnd = startEndCell[gid];

  if (startEnd.x == 1 && startEnd.y == 0) //if cell is empty, stop all work-items
    return;

  unsigned int lid = get_local_id(0);
  unsigned partIndex = gid * get_local_size(0) + lid;

  // Main assumption, we consider there is n particles in each cell with 0 <= n < get_local_size(0)
  // pos size = get_local_size(0)
  if (partIndex <= startEnd.y)
    pos[lid] = position[partIndex];
  else
    pos[lid] = float4(-1000.0f, -1000.0f, -1000.0f, -1000.0f);

  barrier(CLK_LOCAL_MEM_FENCE);
}
*/

__kernel void applyBoidsRulesWithGrid(
    __global float4* position,
    __global float4* velocity,
    __global float4* acc,
    __global uint2* startEndCell,
    float8 params)
{
  unsigned int i = get_global_id(0);

  float4 pos = position[i];
  float4 vel = velocity[i];

  int3 cell3DIndex = getCell3DIndex(pos);

  int count = 0;

  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int x = 0;
  int y = 0;
  int z = 0;
  uint cellIndex = 0;
  uint2 startEndCurrent = (uint2)(0, 0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = cell3DIndex.x + iX;
        y = cell3DIndex.y + iY;
        z = cell3DIndex.z + iZ;

        if (x >= 0 && x < GRID_RES
            && y >= 0 && y < GRID_RES
            && z >= 0 && z < GRID_RES)
        {
          cellIndex = (x * GRID_RES + y) * GRID_RES + z;

          startEndCurrent = startEndCell[cellIndex];
          for (uint e = startEndCurrent.x; e <= startEndCurrent.y; ++e)
          {
            vec = pos - position[e];
            squaredDist = dot(vec, vec);

            // Second condition to deal with almost identical points generated by parallelRNG and i == e
            if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
            {
              averageBoidsPos += position[e];
              averageBoidsVel += velocity[e];
              repulseHeading += vec / squaredDist;
              ++count;
            }
          }
        }
      }
    }
  }

  // params 0 = vel - 1 = cohesion - 2 = alignement - 3 = separation - 4 = target
  if (count != 0)
  {
    // cohesion
    averageBoidsPos /= count;
    averageBoidsPos -= pos;
    averageBoidsPos = normalize(averageBoidsPos) * params.s0;
    // alignment
    averageBoidsVel = normalize(averageBoidsVel) * params.s0;
    // separation
    repulseHeading = normalize(repulseHeading) * params.s0;
  }

  float4 target = -pos;

  acc[i] = steerForce(averageBoidsPos, vel) * params.s1
      + steerForce(averageBoidsVel, vel) * params.s2
      + steerForce(repulseHeading, vel) * params.s3
      + clamp(target, 0.0, normalize(target) * MAX_STEERING) * params.s4;
}

__kernel void fillBoidsTexture(
    __global uint2* startEndCell,
    __global float4* inputBoidsBuffer,
    __write_only image2d_t outputBoidsText)
{
  int iG = get_group_id(0);
  int iL = get_local_id(0);

  uint2 startEndCellIndex = startEndCell[iG];
  uint nbPartInCell = startEndCellIndex.y - startEndCellIndex.x;

  float4 inPart = (float4)(0.0, 0.0, 0.0, -1.0);
  if (iL < nbPartInCell)
    inPart = inputBoidsBuffer[startEndCellIndex.x + iL];

  int2 coords = (int2)(iL, iG);
  write_imagef(outputBoidsText, coords, inPart);
}

__kernel void applyBoidsRulesWithGridAndTex(
    __global float4* position,
    __global float4* velocity,
    __read_only image2d_t posTex,
    __read_only image2d_t velTex,
    __global float4* acc,
    float8 params)
{
  unsigned int i = get_global_id(0);

  float4 pos = position[i];
  float4 vel = velocity[i];

  int3 cell3DIndex = getCell3DIndex(pos);

  int count = 0;

  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int x = 0;
  int y = 0;
  int z = 0;
  uint cellIndex = 0;

  sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = cell3DIndex.x + iX;
        y = cell3DIndex.y + iY;
        z = cell3DIndex.z + iZ;

        if (x < 0 || x >= GRID_RES
            || y < 0 || y >= GRID_RES
            || z < 0 || z >= GRID_RES)
          return;

        cellIndex = (x * GRID_RES + y) * GRID_RES + z;

        for (uint partIndex = 0; partIndex < 200; ++partIndex)
        {
          float4 posN = read_imagef(posTex, samp, (int2)(partIndex, cellIndex));
          float4 velN = read_imagef(velTex, samp, (int2)(partIndex, cellIndex));

          if (isequal(posN.s3, -1.0f))
            continue;

          vec = pos - posN;
          squaredDist = dot(vec, vec);

          // Second condition to deal with almost identical points generated by parallelRNG and i == e
          if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
          {
            averageBoidsPos += posN;
            averageBoidsVel += velN;
            repulseHeading += vec / squaredDist;
            ++count;
          }
        }
      }
    }
  }

  // params 0 = vel - 1 = cohesion - 2 = alignement - 3 = separation - 4 = target
  if (count != 0)
  {
    // cohesion
    averageBoidsPos /= count;
    averageBoidsPos -= pos;
    averageBoidsPos = normalize(averageBoidsPos) * params.s0;
    // alignment
    averageBoidsVel = normalize(averageBoidsVel) * params.s0;
    // separation
    repulseHeading = normalize(repulseHeading) * params.s0;
  }

  float4 target = -pos;

  acc[i] = steerForce(averageBoidsPos, vel) * params.s1
      + steerForce(averageBoidsVel, vel) * params.s2
      + steerForce(repulseHeading, vel) * params.s3
      + clamp(target, 0.0, normalize(target) * MAX_STEERING) * params.s4;
}