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

__kernel void colorVerts(const global float4* pos, global float4* color)
{
  int i = get_global_id(0);
  float4 currPos = fabs(pos[i]);
  color[i] = (float4)(1.0f - currPos.y / 900.f, 0.2f, 0.0f, 1.0f);
}

__kernel void randPosVerts(global float4* pos, global float4* vel, float dim)
{
  unsigned int i = get_global_id(0);

  unsigned int randomIntX = parallelRNG(i);
  unsigned int randomIntY = parallelRNG(i + 1);
  unsigned int randomIntZ = parallelRNG(i + 2);

  float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0f;
  float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0f;
  float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0f;

  float3 randomXYZ = (float3)(x * step(3.0f, dim), y, z);

  pos[i].xyz = clamp(randomXYZ, -ABS_WALL_POS, ABS_WALL_POS);
  pos[i].w = 0.0;

  vel[i].xyz = clamp(randomXYZ, -50.0f, 50.0f);
  vel[i].w = 0.0;
}

inline float4 steerForce(float4 desiredVel, float4 vel)
{
  float4 steerForce = desiredVel - vel;
  if (length(steerForce) > MAX_STEERING)
  {
    steerForce = fast_normalize(steerForce) * MAX_STEERING;
  }
  return steerForce;
}

// For rendering purpose only, checking for each grid cell if there is any particle
// If so, grid cell will be displayed in OpenGl
__kernel void flushGridDetector(global float8* gridDetector)
{
  unsigned int i = get_global_id(0);
  gridDetector[i] = 0.0f;
}

inline int3 getCell3DIndexFromPos(float4 pos)
{
  int cellSize = 2 * ABS_WALL_POS / GRID_RES;

  // Moving particles in [0 - 2 * ABS_WALL_POS] to have coords matching with cellIndices
  // Adding epsilon to avoid wrong indices if particle exactly on the ABS_WALL_POS
  float3 posXYZ = pos.xyz + ABS_WALL_POS - FLOAT_EPSILON;

  int3 cell3DIndex = convert_int3(posXYZ / cellSize);

  clamp(cell3DIndex, 0, GRID_RES - 1);

  return cell3DIndex;
}

inline uint getCell1DIndexFromPos(float4 pos)
{
  int3 cell3DIndex = getCell3DIndexFromPos(pos);

  uint cell1DIndex = cell3DIndex.x * GRID_RES * GRID_RES
      + cell3DIndex.y * GRID_RES
      + cell3DIndex.z;

  return cell1DIndex;
}

__kernel void fillGridDetector(__global float4* vertPos, __global float8* gridDetector)
{
  unsigned int i = get_global_id(0);

  float4 pos = vertPos[i];

  uint gridDetectorIndex = getCell1DIndexFromPos(pos);

  if (gridDetectorIndex < GRID_RES * GRID_RES * GRID_RES) // WIP
    gridDetector[gridDetectorIndex] = 1.0f;
}

// To use of Radix Sort accelerator, we need to find the cellID for each boids particle
__kernel void flushCellIDs(global uint* boidsCellIDs)
{
  unsigned int i = get_global_id(0);
  // For all particles, giving cell ID above any available one
  // the ones not filled later (i.e not processed because index > numEntites displayed)
  // will be sorted at the end and not considered after sorting
  boidsCellIDs[i] = GRID_NUM_CELLS + 1;
}

__kernel void fillCellIDs(const global float4* vertPos, global uint* boidsCellIDs)
{
  unsigned int i = get_global_id(0);

  float4 pos = vertPos[i];

  uint cell1DIndex = getCell1DIndexFromPos(pos);

  boidsCellIDs[i] = cell1DIndex;
}

__kernel void flushStartEndCell(global uint2* startEndCells)
{
  unsigned int i = get_global_id(0);

  // Flushing with 1 as starting index and 0 as ending index
  // Little hack to bypass empty cell further in the boids algo
  startEndCells[i] = (uint2)(1, 0);
}

__kernel void fillStartCell(const global uint* boidsCellIDs, global uint2* startEndCells)
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

__kernel void fillEndCell(const global uint* boidsCellIDs, global uint2* startEndCells)
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

__kernel void adjustEndCell(global uint2* startEndCells)
{
  unsigned int i = get_global_id(0);

  uint2 startEnd = startEndCells[i];
  if (startEnd.y > startEnd.x)
  {
    uint newEnd = startEnd.x + min(startEnd.y - startEnd.x, (uint)NUM_MAX_PARTS_IN_CELL);
    startEndCells[i] = (uint2)(startEnd.x, newEnd);
  }
}

__kernel void applyBoidsRulesWithGrid(
    // global inputs
    const global float4* position,
    const global float4* velocity,
    const global uint2* startEndCell,
    // param
    const float8 params,
    // global output
    global float4* acc)
{
  unsigned int i = get_global_id(0);

  float4 pos = position[i];
  float4 vel = velocity[i];

  uint cell1DIndex = getCell1DIndexFromPos(pos);

  uint2 startEnd = startEndCell[cell1DIndex];

  int count = 0;

  float4 newAcc = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int x = 0;
  int y = 0;
  int z = 0;
  uint cellIndex = 0;
  uint2 startEndN = (uint2)(0, 0);
  int3 currentCell3DIndex = getCell3DIndexFromPos(pos);

  float4 posN = (float4)(0.0, 0.0, 0.0, 0.0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = currentCell3DIndex.x + iX;
        y = currentCell3DIndex.y + iY;
        z = currentCell3DIndex.z + iZ;

        if (x < 0 || x >= GRID_RES
            || y < 0 || y >= GRID_RES
            || z < 0 || z >= GRID_RES)
          continue;

        cellIndex = (x * GRID_RES + y) * GRID_RES + z;

        startEndN = startEndCell[cellIndex];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          posN = position[e];

          vec = pos - posN;
          squaredDist = dot(vec, vec);

          // Second condition to deal with almost identical points generated by parallelRNG and i == e
          if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
          {
            averageBoidsPos += posN;
            averageBoidsVel += velocity[e];
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
    averageBoidsPos = fast_normalize(averageBoidsPos) * params.s0;
    // alignment
    averageBoidsVel = fast_normalize(averageBoidsVel) * params.s0;
    // separation
    repulseHeading = fast_normalize(repulseHeading) * params.s0;

    newAcc = steerForce(averageBoidsPos, vel) * params.s1
        + steerForce(averageBoidsVel, vel) * params.s2
        + steerForce(repulseHeading, vel) * params.s3;
  }

  acc[i] = newAcc;
}

__kernel void addTargetRule(
    // global input
    const global float4* pos,
    // params
    const float4 targetPos,
    const float targetSquaredRadiusEffect,
    const int targetSignEffect,
    // global output
    global float4* acc)
{
  unsigned int i = get_global_id(0);

  float4 currPos = pos[i];

  float4 vec = targetPos - currPos;
  float squaredDist = dot(vec, vec);

  float4 targetAcc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  if (squaredDist < targetSquaredRadiusEffect)
    targetAcc += targetSignEffect * clamp(vec, 0.0, fast_normalize(vec) * MAX_STEERING);

  acc[i] += targetAcc;
}

__kernel void updateVel(
    // global input
    const global float4* acc,
    // params
    const float timeStep,
    const float velocity,
    // global output
    global float4* vel)
{
  unsigned int i = get_global_id(0);

  float4 newVel = vel[i] + acc[i] * timeStep;
  vel[i] = fast_normalize(newVel) * velocity;
}

__kernel void updatePosWithBouncingWalls(
    // global input/output
    global float4* vel,
    // param
    const float timeStep,
    // global output
    global float4* pos)
{
  unsigned int i = get_global_id(0);

  float4 newPos = pos[i] + vel[i] * timeStep;
  float4 clampedNewPos = clamp(newPos, -ABS_WALL_POS, ABS_WALL_POS);
  if (!all(isequal(clampedNewPos.xyz, newPos.xyz)))
  {
    vel[i] *= -0.5f;
  }
  pos[i] = clampedNewPos;
}

__kernel void updatePosWithCyclicWalls(
    // global input
    const global float4* vel,
    // param
    const float timeStep,
    // global output
    global float4* pos)
{
  unsigned int i = get_global_id(0);

  float4 newPos = pos[i] + vel[i] * timeStep;
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

// Classic boids physics with no grid, O(N^2) in time complexity
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

// Implementation of the boids with grid using a texture to store first n pos in each cell
// Big approximation as we only keep the n first ones and, worse, slower than simply using classic global memory!
/*
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

  if (startEndCellIndex.y >= startEndCellIndex.x && iL < nbPartInCell)
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
    __global uint2* startEndCell,
    float8 params)
{
  unsigned int i = get_global_id(0);

  float4 pos = position[i];
  float4 vel = velocity[i];

  uint cell1DIndex = getCell1DIndexFromPos(pos);

  uint2 startEnd = startEndCell[cell1DIndex];

  if ((startEnd.y - startEnd.x) <= 20 * NUM_MAX_PARTS_IN_CELL)
    return;

  int count = 0;

  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int x = 0;
  int y = 0;
  int z = 0;
  int3 currentCell3DIndex = getCell3DIndexFromPos(pos);
  uint cellIndex = 0;

  sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  float4 posN = (float4)(0.0, 0.0, 0.0, 0.0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = currentCell3DIndex.x + iX;
        y = currentCell3DIndex.y + iY;
        z = currentCell3DIndex.z + iZ;

        if (x < 0 || x >= GRID_RES
            || y < 0 || y >= GRID_RES
            || z < 0 || z >= GRID_RES)
          return;

        cellIndex = (x * GRID_RES + y) * GRID_RES + z;

        for (uint partIndex = 0; partIndex < NUM_MAX_PARTS_IN_CELL; ++partIndex)
        {
          posN = read_imagef(posTex, samp, (int2)(partIndex, cellIndex));

          if (isequal(posN.s3, -1.0f))
            continue;

          vec = pos - posN;
          squaredDist = dot(vec, vec);

          // Second condition to deal with almost identical points generated by parallelRNG and i == e
          if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
          {
            averageBoidsPos += posN;
            averageBoidsVel += read_imagef(velTex, samp, (int2)(partIndex, cellIndex));
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

// Tentative to use local/shared memory with boids with grid and texture
// Too demanding on local/shared memory side, as we need to store vel and pos for n pos for 27 cells 
// (float4 * 2 * n * 27), cannot get more than n = 50 with gtx1650...

__kernel void applyBoidsRulesWithGridAndTexLocal(
    __global uint2* startEndCell,
    __read_only image2d_t posTex,
    __read_only image2d_t velTex,
    __global float4* acc,
    float8 params,
    __local float4* localPos,
    __local float4* localVel)
{
  unsigned int cellIndex = get_group_id(0);
  unsigned int localPartIndex = get_local_id(0);

  uint2 startEnd = startEndCell[cellIndex];
  uint nbPartInCell = startEnd.y - startEnd.x;

  if (startEnd.y < startEnd.x)
    return;

  sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  float4 posF = read_imagef(posTex, samp, (int2)(0, cellIndex));
  int3 cell3DIndex = getCell3DIndexFromPos(posF);

  // local filling for the 27 cells

  int x = 0;
  int y = 0;
  int z = 0;
  uint cellIndexN = 0;
  uint localIndexCellN = 0;
  float4 posN = (float4)(0.0, 0.0, 0.0, -1.0);
  float4 velN = (float4)(0.0, 0.0, 0.0, -1.0);

  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = cell3DIndex.x + iX;
        y = cell3DIndex.y + iY;
        z = cell3DIndex.z + iZ;

        posN = (float4)(0.0, 0.0, 0.0, -1.0);
        velN = (float4)(0.0, 0.0, 0.0, -1.0);

        if (x >= 0 && x < GRID_RES
            && y >= 0 && y < GRID_RES
            && z >= 0 && z < GRID_RES)
        {
          cellIndexN = (x * GRID_RES + y) * GRID_RES + z;
          posN = read_imagef(posTex, samp, (int2)(localPartIndex, cellIndexN));
          velN = read_imagef(velTex, samp, (int2)(localPartIndex, cellIndexN));
        }
        localPos[localIndexCellN * NUM_MAX_PARTS_IN_CELL + localPartIndex] = posN;
        localVel[localIndexCellN * NUM_MAX_PARTS_IN_CELL + localPartIndex] = velN;

        ++localIndexCellN;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float4 pos = read_imagef(posTex, samp, (int2)(localPartIndex, cellIndex));
  float4 vel = read_imagef(velTex, samp, (int2)(localPartIndex, cellIndex));

  // Not a real part
  if (isequal(pos.w, (float)(-1.0)))
    return;

  float4 averageBoidsPos = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 averageBoidsVel = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 repulseHeading = (float4)(0.0, 0.0, 0.0, 0.0);
  int count = 0;

  float4 localPosN = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 localVelN = (float4)(0.0, 0.0, 0.0, 0.0);
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  float squaredDist = 0.0f;

  //size_t numLocalParts = 27 * NUM_MAX_PARTS_IN_CELL;
  size_t numLocalParts = 9 * NUM_MAX_PARTS_IN_CELL;
  for (uint i = 0; i < numLocalParts; ++i)
  {
    localPosN = localPos[i];
    localVelN = localVel[i];

    // Not a real neighbor
    if (isequal(localPosN.w, (float)(-1.0)))
      continue;

    vec = pos - localPosN;
    squaredDist = dot(vec, vec);

    // Second condition to deal with almost identical points generated by parallelRNG
    if (squaredDist < EFFECT_RADIUS_SQUARED && squaredDist > FLOAT_EPSILON)
    {
      averageBoidsPos += localPosN;
      averageBoidsVel += localVelN;
      repulseHeading += vec / squaredDist;
      ++count;
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

  acc[startEnd.x + localPartIndex] = steerForce(averageBoidsPos, vel) * params.s1
      + steerForce(averageBoidsVel, vel) * params.s2
      + steerForce(repulseHeading, vel) * params.s3
      + clamp(target, 0.0, normalize(target) * MAX_STEERING) * params.s4;
}
*/