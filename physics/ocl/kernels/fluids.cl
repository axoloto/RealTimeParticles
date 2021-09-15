
// Position based fluids model based on NVIDIA paper 
// Muller and al. 2013. "Position Based Fluids"

// Preprocessor defines following constant variables in Boids.cpp
// EFFECT_RADIUS_SQUARED   - squared radius around a particle where boids laws apply 
// ABS_WALL_POS            - absolute position of the walls in x,y,z
// GRID_RES                - resolution of the grid
// GRID_NUM_CELLS          - total number of cells in the grid
// NUM_MAX_PARTS_IN_CELL   - maximum number of particles taking into account in a single cell in simplified mode
// REST_DENSITY            - rest density of the fluid

#define MAX_STEERING  0.5f
#define FLOAT_EPSILON 0.01f
#define FAR_DIST      100000000.0f
#define ID            get_global_id(0)
#define GRAVITY_ACC   (float4)(0.0f, -9.81f, 0.0f, 0.0f)
/*
  Random number generator
*/
inline unsigned int parallelRNG(unsigned int i)
{
  unsigned int value = i;

  value = (value ^ 61) ^ (value >> 16);
  value *= 9;
  value ^= value << 4;
  value *= 0x27d4eb2d;
  value ^= value >> 15;

  return value;
}

/*
  Reset camera distance buffer
*/
__kernel void resetCameraDist(__global uint *cameraDist)
{
  cameraDist[ID] = (uint)(FAR_DIST * 2);
}

/*
  Fill camera distance buffer
*/
__kernel void fillCameraDist(//Input
                             const __global float4 *pos,          // 0
                             const __global float3 *cameraPos,    // 1
                             //Output
                                   __global uint   *cameraDist)   // 2
{
  cameraDist[ID] = (uint)(FAR_DIST - fast_length(pos[ID].xyz - cameraPos[0].xyz));
}

/*
  Fill position buffer with inf positions
*/
__kernel void infPosVerts(__global float4 *pos)
{
  pos[ID] = (float4)(FAR_DIST, FAR_DIST, FAR_DIST, 0.0f);
}

/*
  Fill position buffer with random positions
*/
__kernel void randPosVerts(//Output
                           __global float4 *pos, 
                           __global float4 *vel,
                           //Param
                                    float  dim)
{
  const unsigned int randomIntX = parallelRNG(ID);
  const unsigned int randomIntY = parallelRNG(ID + 1);
  const unsigned int randomIntZ = parallelRNG(ID + 2);

  const float x = (float)(randomIntX & 0x0ff) * 2.0 - 250.0f;
  const float y = (float)(randomIntY & 0x0ff) * 2.0 - 250.0f;
  const float z = (float)(randomIntZ & 0x0ff) * 2.0 - 250.0f;

  const float3 randomXYZ = (float3)(x * step(3.0f, dim), y, z);

  pos[ID].xyz = clamp(randomXYZ, -ABS_WALL_POS, ABS_WALL_POS);
  pos[ID].w = 0.0f;

  vel[ID].xyz = (float3)(0.0f, 0.0f, 0.0f);
  vel[ID].w = 0.0f;
}

/*
  Compute 3D index of the cell containing given position
*/
inline int3 getCell3DIndexFromPos(float4 pos)
{
  const int cellSize = 2 * ABS_WALL_POS / GRID_RES;

  // Moving particles in [0 - 2 * ABS_WALL_POS] to have coords matching with cellIndices
  // Adding epsilon to avoid wrong indices if particle exactly on the ABS_WALL_POS
  const float3 posXYZ = pos.xyz + ABS_WALL_POS - FLOAT_EPSILON;

  const int3 cell3DIndex = convert_int3(posXYZ / cellSize);

  return cell3DIndex;
}

/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos)
{
  const int3 cell3DIndex = getCell3DIndexFromPos(pos);

  const uint cell1DIndex = cell3DIndex.x * GRID_RES * GRID_RES
                         + cell3DIndex.y * GRID_RES
                         + cell3DIndex.z;

  return cell1DIndex;
}

/*
  Reset grid detector buffer. For rendering purpose only.
*/
__kernel void flushGridDetector(__global float8* gridDetector)
{
  gridDetector[ID] = 0.0f;
}

/*
  Fill grid detector buffer. For rendering purpose only.
*/
__kernel void fillGridDetector(__global float4 *pPos,
                               __global float8 *gridDetector)
{
  const float4 pos = pPos[ID];

  const uint gridDetectorIndex = getCell1DIndexFromPos(pos);

  if (gridDetectorIndex < GRID_NUM_CELLS)
    gridDetector[gridDetectorIndex] = 1.0f;
}

/*
  Reset cellID buffer. For radix sort purpose.
*/
__kernel void resetCellIDs(__global uint *pCellID)
{
  // For all particles, giving cell ID above any available one
  // the ones not filled later (i.e not processed because index > nbParticles displayed)
  // will be sorted at the end and not considered after sorting
  pCellID[ID] = GRID_NUM_CELLS * 2 + ID;
}

/*
  Fill cellID buffer. For radix sort purpose.
*/
__kernel void fillCellIDs(//Input
                          const __global float4 *pPos,
                          //Output
                                __global uint   *pCellID)
{
  const float4 pos = pPos[ID];

  const uint cell1DIndex = getCell1DIndexFromPos(pos);

  pCellID[ID] = cell1DIndex;
}

/*
  Flush startEndPartID buffer for each cell.
*/
__kernel void flushStartEndCell(__global uint2 *cStartEndPartID)
{
  // Flushing with 1 as starting index and 0 as ending index
  // Little hack to bypass empty cell further in the boids algo
  cStartEndPartID[ID] = (uint2)(1, 0);
}

/*
  Find first partID for each cell.
*/
__kernel void fillStartCell(//Input
                            const __global uint  *pCellID,
                            //Output
                                  __global uint2 *cStartEndPartID)
{
  const uint currentCellID = pCellID[ID];

  if (ID > 0 && currentCellID < GRID_NUM_CELLS)
  {
    uint leftCellID = pCellID[ID - 1];
    if (currentCellID != leftCellID)
    {
      // Found start
      cStartEndPartID[currentCellID].x = ID;
    }
  }
}

/*
  Find last partID for each cell.
*/
__kernel void fillEndCell(//Input
                          const __global uint  *pCellID,
                          //Output
                                __global uint2 *cStartEndPartID)
{
  const uint currentCellID = pCellID[ID];

  if (ID != get_global_size(0) && currentCellID < GRID_NUM_CELLS)
  {
    const uint rightCellID = pCellID[ID + 1];
    if (currentCellID != rightCellID)
    {
      // Found end
      cStartEndPartID[currentCellID].y = ID;
    }
  }
}

/* 
  Adjust last partID for each cell, capping it with max number of parts in cell in simplified mode.
*/
__kernel void adjustEndCell(__global uint2 *cStartEndPartID)
{
  const uint2 startEnd = cStartEndPartID[ID];

  if (startEnd.y > startEnd.x)
  {
    const uint newEnd = startEnd.x + min(startEnd.y - startEnd.x, (uint)NUM_MAX_PARTS_IN_CELL);
    cStartEndPartID[ID] = (uint2)(startEnd.x, newEnd);
  }
}

/*
  Predict fluid particle position and update velocity by integrating external forces
*/
__kernel void predictPosition(//Input
                              const __global float4 *pos,        // 0
                              //Input/Output
                                    __global float4 *vel,        // 1
                              //Param
                              const          float  timeStep,    // 2
                              const          float  maxVelocity, // 3
                              //Output
                                    __global float4 *predPos)    // 4
{
  vel[ID] += maxVelocity * GRAVITY_ACC * timeStep;

  predPos[ID] = pos[ID] + vel[ID] * timeStep;
}

/*
  Compute fluid density based on SPH model
  using predicted position and Poly6 kernel presented in 
  Muller and al. 2003. "Particle-based fluid simulation for interactive applications"
*/
__kernel void computeDensity(//Input
                              const __global float4 *predPos,      // 0
                              const __global uint2  *startEndCell, // 1
                              //Output
                                    __global float  *density)      // 2
{
  const float4 pos = predPos[ID];
  const uint currCell1DIndex = getCell1DIndexFromPos(pos);
  const int3 currCell3DIndex = getCell3DIndexFromPos(pos);
  const uint2 startEnd = startEndCell[currCell1DIndex];

  float fluidDensity = 0.0f;

  int count = 0;

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  int x = 0;
  int y = 0;
  int z = 0;
  uint  cellIndex = 0;
  uint2 startEndN = (uint2)(0, 0);

  float4 posN = (float4)(0.0, 0.0, 0.0, 0.0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        x = currCell3DIndex.x + iX;
        y = currCell3DIndex.y + iY;
        z = currCell3DIndex.z + iZ;

        if (x < 0 || x >= GRID_RES
         || y < 0 || y >= GRID_RES
         || z < 0 || z >= GRID_RES)
          continue;

        cellIndex = (x * GRID_RES + y) * GRID_RES + z;

        startEndN = startEndCell[cellIndex];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          posN = predPos[e];

          vec = pos - posN;
          squaredDist = dot(vec, vec);

          // Second condition to deal with almost identical points generated by parallelRNG and i == e
          if (squaredDist < EFFECT_RADIUS_SQUARED
           && squaredDist > FLOAT_EPSILON)
          {
            fluidDensity += 0.1f; // adding contribution Poly6 (vec, h)
            ++count;
          }
        }
      }
    }
  }

  density[ID] = fluidDensity;
}

/*
  Compute Constraint Factor (Lambda)
*/
__kernel void computeConstraintFactor(//Input
                                      const __global float  *density,       // 0
                                      const __global float4 *predPos,       // 1
                                      //Output
                                            __global float  *constFactor)   // 2
{
  constFactor[ID] = 0.0f;
}

/*
  Compute Constraint Correction
*/
__kernel void computeConstraintCorrection(//Input
                                          const __global float  *constFactor,  // 0
                                          const __global uint2  *startEndCell, // 1
                                          const __global float  *predPos,      // 2
                                          //Output
                                                __global float4 *corrPos)      // 3
{
  // Poly6 and spiky kernel for gradient
  corrPos[ID] = (float4)(0.0f,0.0f,0.0f,0.0f);
}

/*
  Correction position using Constraint correction value
*/
__kernel void correctPosition(//Input
                              const __global float4 *corrPos,  // 0
                              //Input/Output
                                    __global float4 *predPos) // 2
{
  predPos[ID] += corrPos[ID];
}

/*
  Update velocity buffer.
*/
__kernel void updateVel(//Input
                        const __global float4 *predPos,    // 0
                        const __global float4 *pos,        // 1
                        //Param
                        const          float  timeStep,    // 2
                        //Output
                              __global float4 *vel)        // 3
   
{
  vel[ID] =  (predPos[ID] - pos[ID]) / timeStep;
}

/*
  Apply Bouncing wall boundary conditions on position and velocity buffers.
*/
__kernel void updatePosWithBouncingWalls(//Input/output
                                               __global float4 *predPos, // 0
                                         //Output
                                               __global float4 *pos)     // 1

{
  pos[ID] = clamp(predPos[ID], -ABS_WALL_POS, ABS_WALL_POS);
}

/*
  Apply Cyclic wall boundary conditions on position and velocity buffers.
*/
__kernel void updatePosWithCyclicWalls(//Input
                                      const __global float4 *predPos, // 0
                                      //Input/output
                                            __global float4 *pos)     // 2
{
  const float4 newPos = predPos[ID];
  const float4 clampedNewPos = clamp(newPos, -ABS_WALL_POS, ABS_WALL_POS);

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

  pos[ID] = clampedNewPos;
}
