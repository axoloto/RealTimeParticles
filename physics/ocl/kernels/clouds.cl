// Clouds simulation model based on 
// "Adaptive cloud simulation using position based fluids", CWF Barbosa, Yoshinori Dobashi & Tsuyoshi Yamamoto 2015.

// Preprocessor defines following constant variables in Clouds.cpp
// EFFECT_RADIUS           - radius around a particle where SPH laws apply 
// ABS_WALL_POS            - absolute position of the walls in x,y,z
// GRID_RES                - resolution of the grid
// GRID_NUM_CELLS          - total number of cells in the grid
// NUM_MAX_PARTS_IN_CELL   - maximum number of particles taking into account in a single cell in simplified mode
// REST_DENSITY            - rest density of the fluid
// POLY6_COEFF             - coefficient of the Poly6 kernel, depending on EFFECT_RADIUS
// SPIKY_COEFF             - coefficient of the Spiky kernel, depending on EFFECT_RADIUS

// Most defines are in define.cl
// define.cl must be included as first file.cl to create OpenCL program
#define WALL_COEFF 1000.0f

// See CloudKernelInputs in Clouds.cpp
typedef struct defCloudParams{
  float timeStep;
  float groundHeatCoeff;
  float buoyancyCoeff;
  float adiabaticLapseRate;
  float phaseTransitionRate;
  float latentHeatCoeff;
  uint  isTempSmoothingEnabled;
  float effectRadius;
  float restDensity;
  float relaxCFM;
} CloudParams;

// Defined in utils.cl
/*
  Random unsigned integer number generator
*/
inline unsigned int parallelRNG(unsigned int i);

// Defined in grid.cl
/*
  Compute 3D index of the cell containing given position
*/
inline uint3 getCell3DIndexFromPos(float4 pos);
/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos);

// Defined in fluids.cl
/*
  Poly6 kernel introduced in
  Muller et al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null value if vec length is superior to effectRadius
*/
inline float poly6(const float4 vec, const float effectRadius);

inline float poly6L(const float vecLength, const float effectRadius);
/*
  Jacobian (on vec coords) of Spiky kernel introduced in
  Muller et al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null vector if vec length is superior to effectRadius
*/
inline float4 gradSpiky(const float4 vec, const float effectRadius);

/*
  Function representing the external heat source at ground level
  Value of 1 at the ground level and then exponentially decreasing
*/
inline float externalHeatSource(const float altitude)
{
  return clamp(exp(-(altitude + ABS_WALL_POS)), 0.0f, 1.0f);
}

/*
  Linear function describing the environment temperature
*/
inline float environmentTemp(const float altitude)
{
  float coeff = 6.0f; // 293 - 6 * 10 = 233K = value of the temperature at the maximum altitude
  float tempGround = 293.0f; // 293K = value of temperature at the ground in Kelvin
  return -coeff * (altitude + ABS_WALL_POS) + tempGround;
}

/*

*/
inline float maxVaporDensity(const float temperature)
{
  // Values given by CWF Barbosa et al. paper
  return 100.0f * exp(-3.0f / (temperature + 2.3f));
}

/*
  Initialize environment temperature field using linear function defined above, same one used for buoyancy computation
*/
__kernel void cld_initTemperature(//Input
                                  const __global float4 *pos,    // 0
                                  //Output
                                        __global float  *temp)   // 1
{
  temp[ID] = environmentTemp(pos[ID].y);
}

/*
  Initialize vapor density value (cloud density set to 0)
*/
 __kernel void cld_initVaporDensity(//Input
                                    const __global float *temp,        // 0
                                    //Output
                                          __global float *vaporDens)   // 1
{
  vaporDens[ID] = 0.5f * maxVaporDensity(temp[ID]);
}

/*
  Fill position buffer with random positions
*/
__kernel void cld_randPosVertsClouds(//Param
                                     const FluidParams fluid, // 0
                                     //Output
                                     __global   float4 *pos,  // 1
                                     __global   float4 *vel)  // 2
{
  const unsigned int randomIntX = parallelRNG(ID);
  const unsigned int randomIntY = parallelRNG(ID + 1);
  const unsigned int randomIntZ = parallelRNG(ID + 2);

  const float x = (float)(randomIntX & 0x0ff) * 2.0 - ABS_WALL_POS;
  const float y = (float)(randomIntY & 0x0ff) * 2.0 - ABS_WALL_POS;
  const float z = (float)(randomIntZ & 0x0ff) * 2.0 - ABS_WALL_POS;

  const float3 randomXYZ = (float3)(x * convert_float(3 - fluid.dim), y, z);

  pos[ID].xyz = clamp(randomXYZ, -ABS_WALL_POS, ABS_WALL_POS);
  pos[ID].w = 0.0f;

  vel[ID].xyz = (float3)(0.0f, 0.0f, 0.0f);
  vel[ID].w = 0.0f;
}

/*

*/
__kernel void cld_heatFromGround(//Input
                                 const __global float  *tempIn, // 0
                                 const __global float4 *pos,    // 1
                                 //Param
                                 const     CloudParams cloud,   // 2
                                 //Output
                                       __global float  *temp)   // 3
{
  temp[ID] = tempIn[ID] + externalHeatSource(pos[ID].y) * cloud.groundHeatCoeff * cloud.timeStep;
}

/*

*/
__kernel void cld_computeBuoyancy(//Input
                                  const __global float  *tempIn,    // 0
                                  const __global float4 *pos,       // 1
                                  const __global float  *cloudDens, // 2
                                  //Param
                                  const     CloudParams cloud,      // 3
                                  //Output
                                        __global float  *buoyancy)  // 4
{
  float envTemp = environmentTemp(pos[ID].y);
  buoyancy[ID] = cloud.buoyancyCoeff * (tempIn[ID] - envTemp) / envTemp - GRAVITY_ACC_Y * cloudDens[ID];
}

/*

*/
__kernel void cld_applyAdiabaticCooling(//Input
                                        const __global float  *tempIn,  // 0
                                        const __global float4 *vel,     // 1
                                        //Param
                                        const     CloudParams cloud,    // 2
                                        //Output
                                              __global float  *temp)    // 3
{
  temp[ID] = tempIn[ID] - cloud.adiabaticLapseRate * max(vel[ID].y, 0.0f) * cloud.timeStep;
}

/*

*/
__kernel void cld_generateCloud(//Input
                                const __global float  *tempIn,    // 0
                                const __global float  *vaporDens, // 1
                                const __global float  *cloudDens, // 2
                                //Param
                                const     CloudParams cloud,      // 3
                                //Output
                                      __global float  *cloudGen)  // 4
{
  cloudGen[ID] = cloud.phaseTransitionRate * (vaporDens[ID] - min(maxVaporDensity(tempIn[ID]), cloudDens[ID] + vaporDens[ID]));
}

/*

*/
__kernel void cld_applyPhaseTransition(//Input
                                       const __global float  *vaporDensIn,  // 0
                                       const __global float  *cloudDensIn,  // 1
                                       const __global float  *cloudGen,     // 2
                                       //Param
                                       const     CloudParams cloud,         // 3
                                       //Output
                                             __global float  *vaporDens,    // 4
                                             __global float  *cloudDens)    // 5
{
  cloudDens[ID] = cloudDensIn[ID] + cloudGen[ID] * cloud.timeStep;
  vaporDens[ID] = vaporDensIn[ID] - cloudGen[ID] * cloud.timeStep;
}

/*

*/
__kernel void cld_applyLatentHeat(//Input
                                  const __global float  *tempIn,   // 0
                                  const __global float  *cloudGen, // 1
                                  //Param
                                  const     CloudParams cloud,     // 2
                                  //Output
                                        __global float  *temp)     // 3
{
  temp[ID] = tempIn[ID] + cloud.latentHeatCoeff * cloudGen[ID] * cloud.timeStep;
}

/*
  Predict fluid particle position and update velocity by integrating external forces
*/
__kernel void cld_predictPosition(//Input
                                  const __global float4 *pos,        // 0
                                  const __global float4 *vel,        // 1
                                  const __global float  *buoyancy,   // 2
                                  //Param
                                  const     CloudParams cloud,       // 3
                                  //Output
                                        __global float4 *predPos)    // 4
{
  // No need to update global vel, as it will be reset later on
  const float4 newVel = vel[ID] + (float4)(0.0f, buoyancy[ID], 0.0f, 0.0f) * cloud.timeStep;

  predPos[ID] = pos[ID] + newVel * cloud.timeStep;
}

/*
  Fill fluid color buffer with constraint value for real-time analysis
  Blue => constraint == 0, i.e density close from the rest density, system close from equilibrium
  Light blue => constraint > 0, i.e density is smaller than rest density, system is not stabilized
  Dark blue => constraint < 0, i.e density is bigger than rest density, system is not stabilized
*/
__kernel void cld_fillCloudColor(//Input
                                 const  __global float  *density, // 0
                                 const  __global float  *densityV, // 1
                                 //Param
                                 const      FluidParams fluid,    // 2
                                 //Output
                                        __global float4 *col)     // 3
{
  float4 blue      = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
  float4 lightBlue = (float4)(0.7f, 0.7f, 1.0f, 1.0f);
  float4 darkBlue  = (float4)(0.0f, 0.0f, 0.8f, 1.0f);

  //float constraint = (1.0f - density[ID] / fluid.restDensity);
  //float constraint =  (density[ID] - 233) / 60;
  float constraint =  (density[ID]);
  //float constraint = density[ID];

  float4 color = blue;
  color.x = constraint * 3;
  color.y = densityV[ID] / 100;

 // if(constraint > 0.0f)
 //   color += constraint * (lightBlue - blue) / 0.35f;
 // else if(constraint < 0.0f)
 //   color += constraint * (blue - darkBlue) / 0.35f;

  col[ID] = color;
}

/*
  Apply Cyclic wall boundary conditions for xz directions
  Apply Boucing wall boundary conditions for y direction
*/
__kernel void cld_applyBoundaryCondWithMixedWalls(//Input/output
                                                  __global float4 *predPos // 0
                                                  )     // 1
{
  float4 newPos = predPos[ID];
  float4 clampedNewPos = clamp(newPos, -ABS_WALL_POS, ABS_WALL_POS);

  if (!isequal(clampedNewPos.x, newPos.x))
  {
    predPos[ID].x = -clampedNewPos.x;
  }
  if (!isequal(clampedNewPos.y, newPos.y))
  {
    predPos[ID].y = clampedNewPos.y;
  }  
  if (!isequal(clampedNewPos.z, newPos.z))
  {
    predPos[ID].z = -clampedNewPos.z;
  }
}


//
//Compute Laplacian temperature based on SPH model
//using position and gradSpiky kernel
//
__kernel void cld_computeLaplacianTemp(//Input
                                       const __global float4 *posP,           // 0
                                       const __global float  *tempP,          // 1
                                       const __global uint2  *startEndCell,   // 2
                                       //Param
                                       const     CloudParams cloud,           // 3
                                       //Output
                                             __global float  *laplacianTemp)  // 4
{
  const float4 pos = posP[ID];
  const float temp = tempP[ID];
  const uint3 cellIndex3D = getCell3DIndexFromPos(pos);

  float laplacian = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  uint2 startEndN = (uint2)(0, 0);
  float4 vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        cellNIndex3D = convert_int3(cellIndex3D) + (int3)(iX, iY, iZ);

        // Removing out of range cells
        if(any(cellNIndex3D < (int3)(0)) || any(cellNIndex3D >= (int3)(GRID_RES)))
          continue;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES + cellNIndex3D.y) * GRID_RES + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e];

          laplacian += (temp - tempP[e]) * dot(vec, gradSpiky(vec, cloud.effectRadius)) / (dot(vec, vec) + FLOAT_EPS);
        }
      }
    }
  }

  laplacianTemp[ID] = laplacian / cloud.restDensity;
}


//
//Compute Constraint Factor (Lambda), coefficient along jacobian
//
__kernel void cld_computeConstraintFactor(//Input
                                          const __global float4 *posP,             // 0
                                          const __global float  *laplacianTemp,    // 1
                                          const __global uint2  *startEndCell,     // 2
                                          //Param
                                          const     CloudParams cloud,             // 3
                                          //Output
                                                __global float  *constFactorTemp)  // 4
{
  const float4 pos = posP[ID];
  const float laplacian = laplacianTemp[ID];
  const uint3 cellIndex3D = getCell3DIndexFromPos(pos);

  float4 vec = (float4)(0.0f);
  float4 grad = (float4)(0.0f);
  float derivativeTemp = 0.0f;
  float sumGradCi = 0.0f;
  float  sumSqGradC = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  uint2 startEndN = (uint2)(0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        cellNIndex3D = convert_int3(cellIndex3D) + (int3)(iX, iY, iZ);

        // Removing out of range cells
        if(any(cellNIndex3D < (int3)(0)) || any(cellNIndex3D >= (int3)(GRID_RES)))
          continue;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES + cellNIndex3D.y) * GRID_RES + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e];

          // Supposed to be null if vec = 0.0f;
          grad = gradSpiky(vec, cloud.effectRadius);

          derivativeTemp = dot(vec, grad) / (dot(vec, vec) * cloud.restDensity + FLOAT_EPS);
          // Contribution from the ID particle
          sumGradCi += derivativeTemp;
          // Contribution from its neighbors
          sumSqGradC += derivativeTemp * derivativeTemp;
        }
      }
    }
  }

  sumSqGradC += sumGradCi * sumGradCi;

  constFactorTemp[ID] = - laplacian / (sumSqGradC + cloud.relaxCFM);
}


//
//Compute Constraint Correction for temperature
//
__kernel void cld_computeConstraintCorrection(//Input
                                              const __global float  *constFactorTemp,// 0
                                              const __global uint2  *startEndCell,   // 1
                                              const __global float4 *posP,           // 2
                                              //Param
                                              const     CloudParams cloud,           // 3
                                              //Output
                                                    __global float *corrTemp)       // 4
{
  const float4 pos = posP[ID];
  const float lambdaI = constFactorTemp[ID];
  const uint3 cellIndex3D = getCell3DIndexFromPos(pos);

  float4 vec = (float4)(0.0f);
  float4 grad = (float4)(0.0f);
  float derivativeTemp = 0.0f;
  float corr = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  uint2 startEndN = (uint2)(0, 0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        cellNIndex3D = convert_int3(cellIndex3D) + (int3)(iX, iY, iZ);

        // Removing out of range cells
        if(any(cellNIndex3D < (int3)(0)) || any(cellNIndex3D >= (int3)(GRID_RES)))
          continue;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES + cellNIndex3D.y) * GRID_RES + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e];
          
          // Supposed to be null if vec = 0.0f;
          grad = gradSpiky(vec, cloud.effectRadius);

          derivativeTemp = dot(vec, grad) / (dot(vec, vec) * cloud.restDensity + FLOAT_EPS);

          corr += (lambdaI + constFactorTemp[e]) * derivativeTemp;
        }
      }
    }
  }

  corrTemp[ID] = corr;
}


//
// Correction on temperature field using Constraint correction value
//
__kernel void cld_correctTemperature(//Input
                                     const __global float *corrTemp, // 0
                                     //Output
                                           __global float *temp)     // 2
{
  temp[ID] += 0.3f * corrTemp[ID];
}
