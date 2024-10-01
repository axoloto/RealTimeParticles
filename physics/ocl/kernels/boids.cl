// Preprocessor defines following constant variables in Boids.cpp
// EFFECT_RADIUS_SQUARED   - squared radius around a particle where boids laws apply 
// ABS_WALL_X            - absolute position of the walls in x,y,z
// GRID_RES_X                - resolution of the grid
// GRID_NUM_CELLS          - total number of cells in the grid
// NUM_MAX_PARTS_IN_CELL   - maximum number of particles taking into account in a single cell in simplified mode

// Most defines are in define.cl
// define.cl must be included as the first .cl file to create the OpenCL program 
#define MAX_STEERING  0.5f

typedef struct defBoidsRuleParams{
  float velocityScale;
  float alignmentScale;
  float separationScale;
  float cohesionScale;
} BoidsRuleParams;

typedef struct defTargetParams{
  float targetRadiusEffect;
  int targetSignEffect;
} TargetParams;

// Defined in grid.cl
/*
  Compute 3D index of the cell containing given position
*/
inline uint3 getCell3DIndexFromPos(float4 pos);
/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos);


/*
  Fill color buffer with chosen color, could be a physics variable for real-time analysis
*/
__kernel void bd_fillBoidsColor(__global float4 *col)
{
  col[ID] = (float4)(1.0f, 0.02f, 0.02f, 0.5f);
}

/*
  Apply 3 boids rules using grid in 3D.
*/
__kernel void bd_applyBoidsRulesWithGrid3D(//Input
                                           const __global float4 *position,     // 0
                                           const __global float4 *velocity,     // 1
                                           const __global uint2  *startEndCell, // 2
                                           //Param
                                           const  BoidsRuleParams params,       // 3
                                           //Output
                                                 __global float4 *acc)          // 4
{
  const float4 pos = position[ID];
  const float4 vel = velocity[ID];

  const uint currCellIndex1D = getCell1DIndexFromPos(pos);
  const uint3 currCellIndex3D = getCell3DIndexFromPos(pos);
  const uint2 startEnd = startEndCell[currCellIndex1D];

  int count = 0;

  float4 newAcc = (float4)(0.0f);
  float4 averageBoidsPos = (float4)(0.0f);
  float4 averageBoidsVel = (float4)(0.0f);
  float4 repulseHeading  = (float4)(0.0f);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  uint2 startEndN = (uint2)(0);
  float4 posN = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        cellNIndex3D = convert_int3(currCellIndex3D) + (int3)(iX, iY, iZ);
        
        // Removing out of range cells
        if(any(cellNIndex3D < (int3)(0)) || any(cellNIndex3D >= (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z)))
          continue;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          posN = position[e];

          vec = pos - posN;
          squaredDist = dot(vec, vec);

          // Second condition to deal with almost identical points and i == e
          if (squaredDist < EFFECT_RADIUS_SQUARED
           && squaredDist > FLOAT_EPS)
          {
            averageBoidsPos += posN;
            averageBoidsVel += fast_normalize(velocity[e]);
            repulseHeading  += vec / squaredDist;
            ++count;
          }
        }
      }
    }
  }

  if (count != 0)
  {
    // cohesion
    averageBoidsPos /= count;
    averageBoidsPos -= pos;
    averageBoidsPos  = fast_normalize(averageBoidsPos) * params.velocityScale;
    // alignment
    averageBoidsVel  = fast_normalize(averageBoidsVel) * params.velocityScale;
    // separation
    repulseHeading   = fast_normalize(repulseHeading)  * params.velocityScale;

    newAcc = averageBoidsVel * params.alignmentScale
           + repulseHeading  * params.separationScale
           + averageBoidsPos * params.cohesionScale;
  }

  acc[ID] = newAcc;
}

/*
  Apply 3 boids rules using grid in 2D.
*/
__kernel void bd_applyBoidsRulesWithGrid2D(//Input
                                           const __global float4 *position,     // 0
                                           const __global float4 *velocity,     // 1
                                           const __global uint2  *startEndCell, // 2
                                           //Param
                                           const  BoidsRuleParams params,       // 3
                                           //Output
                                                 __global float4 *acc)          // 4

{
  const float4 pos = position[ID];
  const float4 vel = velocity[ID];

  const uint currCellIndex1D = getCell1DIndexFromPos(pos);
  const uint3 currCellIndex3D = getCell3DIndexFromPos(pos);
  const uint2 startEnd = startEndCell[currCellIndex1D];

  int count = 0;

  float4 newAcc = (float4)(0.0f);
  float4 averageBoidsPos = (float4)(0.0f);
  float4 averageBoidsVel = (float4)(0.0f);
  float4 repulseHeading  = (float4)(0.0f);

  float squaredDist = 0.0f;
  float4 vec = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  uint2 startEndN = (uint2)(0);
  float4 posN = (float4)(0.0f);

  // 9 cells to visit, current one + 2D YZ neighbors
  for (int iY = -1; iY <= 1; ++iY)
  {
    for (int iZ = -1; iZ <= 1; ++iZ)
    {
      cellNIndex3D = convert_int3(currCellIndex3D) + (int3)(0, iY, iZ);
      
      // Removing out of range cells
      if(any(cellNIndex3D < (int3)(0)) || any(cellNIndex3D >= (int3)(GRID_RES_X)))
        continue;

      cellNIndex1D = (GRID_RES_X / 2 * GRID_RES_X + cellNIndex3D.y) * GRID_RES_Y + cellNIndex3D.z;

      startEndN = startEndCell[cellNIndex1D];

      for (uint e = startEndN.x; e <= startEndN.y; ++e)
      {
        posN = position[e];

        vec = pos - posN;
        squaredDist = dot(vec, vec);

        // Second condition to deal with almost identical points and i == e
        if (squaredDist < EFFECT_RADIUS_SQUARED
         && squaredDist > FLOAT_EPS)
        {
          averageBoidsPos += posN;
          averageBoidsVel += fast_normalize(velocity[e]);
          repulseHeading += vec / squaredDist;
          ++count;
        }
      }
    }
  }

  if (count != 0)
  {
    // cohesion
    averageBoidsPos /= count;
    averageBoidsPos -= pos;
    averageBoidsPos  = fast_normalize(averageBoidsPos) * params.velocityScale;
    // alignment
    averageBoidsVel  = fast_normalize(averageBoidsVel) * params.velocityScale;
    // separation
    repulseHeading   = fast_normalize(repulseHeading)  * params.velocityScale;

    newAcc = averageBoidsVel * params.alignmentScale
           + repulseHeading  * params.separationScale
           + averageBoidsPos * params.cohesionScale;
  }

  acc[ID] = newAcc;
}

/*
  Add target rule.
*/
__kernel void bd_addTargetRule(//Input
                               const __global float4 *pos,      // 0                              
                               //Param
                               const          float4 targetPos, // 1
                               const     TargetParams params,   // 2
                               //Output
                                     __global float4 *acc)      // 3
{
  const float4 currPos = pos[ID];

  const float4 vec = targetPos - currPos;
  const float  dist = fast_length(vec);

  if (dist < params.targetRadiusEffect)
    acc[ID] += params.targetSignEffect * vec * clamp(1.3f / dist, 0.0f, 1.4f * MAX_STEERING);
}

/*
  Update velocity buffer.
*/
__kernel void bd_updateVel(//Input
                           const __global float4 *acc,        // 0
                           //Param
                           const          float  timeStep,    // 1
                           const          float  maxVelocity, // 2
                           //Output
                                 __global float4 *vel)        // 3
   
{
  const float4 newVel = vel[ID] + acc[ID] * timeStep;
  const float  newVelNorm = clamp(fast_length(newVel), 0.2f * maxVelocity, maxVelocity);
  
  vel[ID] = fast_normalize(newVel) * newVelNorm;
}

/*
  Update position and apply wall boundary conditions on position and velocity.
*/
__kernel void bd_updatePosAndApplyWallBC(//Input/output
                                         __global float4 *vel,     // 0
                                         //Param
                                         const    float  timeStep, // 1
                                         //Input/output
                                         __global float4 *pos)     // 2

{
  const float4 newPos = pos[ID] + vel[ID] * timeStep;

  const float4 clampedNewPos = clamp(newPos, (float4)(-ABS_WALL_X, -ABS_WALL_Y, -ABS_WALL_Z, 0.0f),
                                             (float4)( ABS_WALL_X,  ABS_WALL_Y,  ABS_WALL_Z, 0.0f));
  
  pos[ID] = clampedNewPos;

  // Bouncing particle will have its velocity reversed and divided by half
  if (!all(isequal(clampedNewPos.xyz, newPos.xyz)))
  {
    vel[ID] *= -0.5f;
  }
}

/*
  Update position and apply periodic boundary conditions.
*/
__kernel void bd_updatePosAndApplyPeriodicBC(//Input
                                             const __global float4 *vel,     // 0
                                             //Param
                                             const          float  timeStep, // 1
                                             //Input/output
                                                   __global float4 *pos)     // 2
{
  const float4 newPos = pos[ID] + vel[ID] * timeStep;

  float4 clampedNewPos = clamp(newPos, (float4)(-ABS_WALL_X, -ABS_WALL_Y, -ABS_WALL_Z, 0.0f),
                                       (float4)( ABS_WALL_X,  ABS_WALL_Y,  ABS_WALL_Z, 0.0f));
 
  if (!isequal(clampedNewPos.x, newPos.x))
  {
    clampedNewPos.x *= -1.0f;
  }
  if (!isequal(clampedNewPos.y, newPos.y))
  {
    clampedNewPos.y *= -1.0f;
  }
  if (!isequal(clampedNewPos.z, newPos.z))
  {
    clampedNewPos.z *= -1.0f;
  }

  pos[ID] = clampedNewPos;
}