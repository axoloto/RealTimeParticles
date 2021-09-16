
// Position based fluids model based on NVIDIA paper 
// Muller and al. 2013. "Position Based Fluids"

// Preprocessor defines following constant variables in Boids.cpp
// EFFECT_RADIUS           - radius around a particle where boids laws apply 
// ABS_WALL_POS            - absolute position of the walls in x,y,z
// GRID_RES                - resolution of the grid
// GRID_NUM_CELLS          - total number of cells in the grid
// NUM_MAX_PARTS_IN_CELL   - maximum number of particles taking into account in a single cell in simplified mode
// REST_DENSITY            - rest density of the fluid
// POLY6_COEFF             - coefficient of the Poly6 kernel, depending on EFFECT_RADIUS
// SPIKY_COEFF             - coefficient of the Spiky kernel, depending on EFFECT_RADIUS

#define MAX_STEERING  0.5f
#define FLOAT_EPSILON 0.01f
#define ID            get_global_id(0)
#define GRAVITY_ACC   (float4)(0.0f, -9.81f, 0.0f, 0.0f)

// Defined in utils.cl
/*
  Random unsigned integer number generator
*/
inline unsigned int parallelRNG(unsigned int i);

// Defined in grid.cl
/*
  Compute 3D index of the cell containing given position
*/
inline int3 getCell3DIndexFromPos(float4 pos);
/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos);

/*
  Poly6 kernel introduced in
  Muller and al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null value if vec length is superior to effectRadius
*/
inline float poly6(const float4 vec, const float effectRadius)
{
  float vecLength = fast_length(vec);
  return step(effectRadius, vecLength) * POLY6_COEFF * pow((effectRadius * effectRadius - vecLength * vecLength),3);
}

/*
  Spiky kernel introduced in
  Muller and al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null value if vec length is superior to effectRadius
*/
inline float gradSpiky(const float4 vec, const float effectRadius)
{
  float vecLength = fast_length(vec);
  return step(effectRadius, vecLength) * SPIKY_COEFF * -3 * pow((effectRadius - vecLength), 2);
}

/*
  Fill position buffer with random positions
*/
__kernel void randPosVertsFluid(//Output
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
  using predicted position and Poly6 kernel
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

  int x = 0;
  int y = 0;
  int z = 0;
  uint  cellIndex = 0;
  uint2 startEndN = (uint2)(0, 0);

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
          fluidDensity += poly6(pos - predPos[e], (float)EFFECT_RADIUS);
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
  //constFactor[ID] = 0.0f;
  constFactor[ID] = density[ID];
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
 // corrPos[ID] = (float4)(constFactor[ID], 0.0f,0.0f,0.0f);
}

/*
  Correction position using Constraint correction value
*/
__kernel void correctPosition(//Input
                              const __global float4 *corrPos,  // 0
                              //Input/Output
                                    __global float4 *predPos) // 2
{
  //predPos[ID] += 0.0f* corrPos[ID];
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
