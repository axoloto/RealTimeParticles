// Clouds simulation model based on 
// "Adaptive cloud simulation using position based fluids", CWF Barbosa, Yoshinori Dobashi & Tsuyoshi Yamamoto 2015.

// Preprocessor defines following constant variables in Clouds.cpp
// EFFECT_RADIUS           - radius around a particle where SPH laws apply 
// ABS_WALL_X            - absolute position of the walls in x,y,z
// GRID_RES_X                - resolution of the grid
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
  uint dim;
  float timeStep;
  float restDensity;  
  float groundHeatCoeff;
  float buoyancyCoeff;
  float gravCoeff;
  float adiabaticLapseRate;
  float phaseTransitionRate;
  float latentHeatCoeff;
  uint  isTempSmoothingEnabled;
  float relaxCFM;
  float initVaporDensityCoeff;
  float windCoeff;
} CloudParams;

// Defined in utils.cl
inline float genRandomNormalizedFloat(unsigned int i);
inline float3 genRandomNormalizedFloat3(unsigned int i);
//

// Defined in sph.cl
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
//

// Defined in grid.cl
/*
  Compute 3D index of the cell containing given position
*/
inline uint3 getCell3DIndexFromPos(float4 pos);
/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos);
//

/*
  Function representing the external heat source at ground level
  Value of 1 at the ground level and then exponentially decreasing
*/
inline float externalHeatSource(const float altitude)
{
  return clamp(exp(-(altitude + ABS_WALL_Y) / 3.0f), 0.0f, 1.0f);
}

/*
  Linear function describing the environment temperature
*/
inline float environmentTemp(const float altitude)
{
  // Troposphere is the first layer of the atmosphere, high of 10kms
  // this is where lives clouds up to the tropopause
  // The temperature within the troposphere varies linearly
  // between around 293K = 20C at alt 0km and 223k = -50C at alt 10km
  float coeff = 3.5f; // 293 - 3.5 * 20 = 223K = -50C = value of the temperature at the maximum altitude
  float tempGround = 293.0f; // 293K = 20C = value of temperature at the ground
  return -coeff * (altitude + ABS_WALL_Y) + tempGround;
}

/*

*/
inline float saturationVaporDensity(const float temperature)
{
  // Values given by CWF Barbosa et al. paper
 // return 100.0f * exp(-3.0f / (temperature + 2.3f)); // this one doesn't make any sense
  return 217 * exp(19.5 - 4303.4/(temperature - 29.5)) / temperature;
}

/*
  Artificial pressure to remove tensile instability
  Preventing particle clustering and improving surface tension
*/
inline float artPressure(const float4 vec, FluidParams fluid)
{
  if(fluid.isArtPressureEnabled == 0)
    return 0.0f;

  return - fluid.artPressureCoeff * pow((poly6(vec, EFFECT_RADIUS) / poly6L(fluid.artPressureRadius * EFFECT_RADIUS, EFFECT_RADIUS)), fluid.artPressureExp);
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
 __kernel void cld_initVaporDensity(//Param
                                    const CloudParams cloud, // 0
                                    //Input
                                    const __global float *temp,        // 1
                                    //Output
                                          __global float *vaporDens)   // 2
{
  vaporDens[ID] = cloud.initVaporDensityCoeff * saturationVaporDensity(temp[ID]);
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
  const float3 randNormFloat3 = genRandomNormalizedFloat3(ID);
  
  pos[ID].xyz = (float3)(ABS_WALL_X, ABS_WALL_Y / 2.0f, ABS_WALL_Z) * (2.0f * randNormFloat3 - 1.0f);  
  pos[ID].y -= ABS_WALL_Y / 2.0f;
  pos[ID].w = 0.0f;

  vel[ID].xyz = 0.5f * randNormFloat3;
  vel[ID].w = 0.0f;
}

/*
  Heat up air parcels from the ground up to 313K = 40C
*/
__kernel void cld_heatFromGround(//Input
                                 const __global float  *tempIn, // 0
                                 const __global float4 *pos,    // 1
                                 //Param
                                 const     CloudParams cloud,   // 2
                                 //Output
                                       __global float  *temp)   // 3
{
  temp[ID] = min(tempIn[ID] + externalHeatSource(pos[ID].y) * cloud.groundHeatCoeff * cloud.timeStep, 313.0f);
}

/*
  Compute buoyancy and gravitational force
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
  buoyancy[ID] = cloud.buoyancyCoeff * (tempIn[ID] - envTemp) / envTemp - cloud.gravCoeff * ABS_GRAVITY_ACC_Y * cloudDens[ID];
}

/*
  Cool down parcels of air when they go up, minimal temperature is the one at the tropopause 223K = -50C
*/
__kernel void cld_applyAdiabaticCooling(//Input
                                        const __global float  *tempIn,  // 0
                                        const __global float4 *vel,     // 1
                                        //Param
                                        const     CloudParams cloud,    // 2
                                        //Output
                                              __global float  *temp)    // 3
{
  temp[ID] = max(tempIn[ID] - cloud.adiabaticLapseRate * vel[ID].y * cloud.timeStep, 223.0f);
}

/*
  Compute cloud generation from the transition rate and the difference between saturation vapor density and current vapor density
  - If vapor density (VD) == saturation vapor density (SVD), no transition, system is at equilibrium
  - If VD > SVD, not enough vapor in the air, clouds disappear, liquid/droplets transitions to vapor
  - If VD < SVD, too much vapor in the air, clouds appear, vapor transitions to liquid/droplets
  Note that saturation vapor density depends on temperature, cold air can't contain as much vapor as warmer one
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
  cloudGen[ID] = cloud.phaseTransitionRate * (vaporDens[ID] - saturationVaporDensity(tempIn[ID]));
 // Article CWF Barbora use this formula which doesn't make much sense,
 // previous article Miyazaki Dobashi 2002 uses max instead of min
 // and the initial one Miyazaki Dobashi 2001 doesn't use neither max or min...
 // cloudGen[ID] = cloud.phaseTransitionRate * (vaporDens[ID] - min(saturationVaporDensity(tempIn[ID]), cloudDens[ID] + vaporDens[ID]));
}

/*
  Applying phase transition rate to compute new phase density in air parcels, density can't be negative
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
  cloudDens[ID] = max(cloudDensIn[ID] + cloudGen[ID] * cloud.timeStep, 0.0f);
  vaporDens[ID] = max(vaporDensIn[ID] - cloudGen[ID] * cloud.timeStep, 0.0f);
}

/*
  Cloud generation generates latent heat, making the air parcels going up some more
*/
__kernel void cld_applyLatentHeat(//Input
                                  const __global float  *tempIn,   // 0
                                  const __global float  *cloudGen, // 1
                                  //Param
                                  const     CloudParams cloud,     // 2
                                  //Output
                                        __global float  *temp)     // 3
{
  temp[ID] = tempIn[ID] + max(cloud.latentHeatCoeff * cloudGen[ID] * cloud.timeStep, 0.0f);
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
                                        __global float4 *predPos,     // 4
                                        __global float4 *totCorrPos)    // 5
{
  // No need to update global vel, as it will be reset later on
  const float4 predVel = vel[ID] + (float4)(0.0f, buoyancy[ID], 0.0f, 0.0f) * cloud.timeStep;
  
  totCorrPos[ID] = predVel * cloud.timeStep;

  predPos[ID] = pos[ID] + predVel * cloud.timeStep;
}

/*
  Apply periodic boundary conditions for xz directions
  Apply wall boundary conditions for y direction
*/
__kernel void cld_applyMixedBoundaryConditions(//Input/output
                                               __global float4 *predPos) // 0
{
  const float4 newPos = predPos[ID];

  const float4 clampedNewPos = clamp(newPos, (float4)(-ABS_WALL_X, -ABS_WALL_Y, -ABS_WALL_Z, 0.0f),
                                             (float4)( ABS_WALL_X,  ABS_WALL_Y,  ABS_WALL_Z, 0.0f));
  
  if (fabs(newPos.x) > ABS_WALL_X)
  {
    predPos[ID].x = newPos.x - 2 * clampedNewPos.x;
  }
  if (fabs(newPos.y) > ABS_WALL_Y)
  {
    predPos[ID].y = clampedNewPos.y;
  }  
  if (fabs(newPos.z) > ABS_WALL_Z)
  {
    predPos[ID].z = newPos.z - 2 * clampedNewPos.z;
  }
}

/*
  Compute fluid density based on SPH model
  using predicted position and Poly6 kernel
*/
__kernel void cld_computeDensity(//Input
                                 const __global float4 *predPos,      // 0
                                 const __global uint2  *startEndCell, // 1
                                 //Param
                                 const     FluidParams fluid,         // 2
                                 //Output
                                       __global float  *density)      // 3
{
  const float4 pos = predPos[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float fluidDensity = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);
  uint2 startEndN = (uint2)(0, 0);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          fluidDensity += poly6(pos - predPos[e] - absWallXYZ * signAbsWall, EFFECT_RADIUS);
        }
      }
    }
  }

  density[ID] = fluidDensity;
}

/*
  Compute Constraint Factor (Lambda) for fluids, coefficient along jacobian
*/
__kernel void cld_computeConstraintFactor(//Input
                                          const __global float4 *predPos,       // 0
                                          const __global float  *density,       // 1
                                          const __global uint2  *startEndCell,  // 2
                                          //Param
                                          const     FluidParams fluid,          // 3
                                          //Output
                                                __global float  *constFactor)   // 4
{
  const float4 pos = predPos[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));
  const float densityC = density[ID] / fluid.restDensity - 1.0f;

  float4 vec = (float4)(0.0f);
  float4 grad = (float4)(0.0f);
  float4 sumGradCi = (float4)(0.0f);
  float  sumSqGradC = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);
   
  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - predPos[e] - absWallXYZ * signAbsWall;

          // Supposed to be null if vec = 0.0f;
          grad = gradSpiky(vec, EFFECT_RADIUS);
          // Contribution from the ID particle
          sumGradCi += grad;
          // Contribution from its neighbors
          sumSqGradC += dot(grad, grad);
        }
      }
    }
  }

  sumSqGradC += dot(sumGradCi, sumGradCi);
  sumSqGradC /= fluid.restDensity * fluid.restDensity;

  constFactor[ID] = - densityC / (sumSqGradC + fluid.relaxCFM);
}

/*
  Compute Constraint Correction
*/
__kernel void cld_computeConstraintCorrection(//Input
                                              const __global float  *constFactor,  // 0
                                              const __global uint2  *startEndCell, // 1
                                              const __global float4 *predPos,      // 2
                                              //Param
                                              const     FluidParams fluid,         // 3
                                              //Output
                                                    __global float4 *corrPos)      // 4
{
  const float4 pos = predPos[ID];
  const float lambdaI = constFactor[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float4 vec = (float4)(0.0f);
  float4 corr = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - predPos[e] - absWallXYZ * signAbsWall;

          corr += (lambdaI + constFactor[e] + artPressure(vec, fluid)) * gradSpiky(vec, EFFECT_RADIUS);
        }
      }
    }
  }

  corrPos[ID] = corr / fluid.restDensity;
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
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));
  
  float4 vec = (float4)(0.0f);
  float laplacian = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e] - absWallXYZ * signAbsWall;

          laplacian += (temp - tempP[e]) * dot(vec, gradSpiky(vec, EFFECT_RADIUS)) / (dot(vec, vec) + FLOAT_EPS);
        }
      }
    }
  }

  laplacianTemp[ID] = laplacian / cloud.restDensity;
}


//
//Compute Constraint Factor (Lambda) for Temperature field, coefficient along jacobian
//
__kernel void cld_computeConstraintFactorTemp(//Input
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
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float4 vec = (float4)(0.0f);
  float4 grad = (float4)(0.0f);
  float derivativeTemp = 0.0f;
  float sumGradCi = 0.0f;
  float sumSqGradC = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {        
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e] - absWallXYZ * signAbsWall;

          // Supposed to be null if vec = 0.0f;
          grad = gradSpiky(vec, EFFECT_RADIUS);

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
__kernel void cld_computeConstraintCorrectionTemp(//Input
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
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float4 vec = (float4)(0.0f);
  float4 grad = (float4)(0.0f);
  float derivativeTemp = 0.0f;
  float corr = 0.0f;

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);
  
  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vec = pos - posP[e] - absWallXYZ * signAbsWall;
          
          // Supposed to be null if vec = 0.0f;
          grad = gradSpiky(vec, EFFECT_RADIUS);

          derivativeTemp = dot(vec, grad) / (dot(vec, vec) * cloud.restDensity + FLOAT_EPS);

          corr += (lambdaI + constFactorTemp[e]) * derivativeTemp;
        }
      }
    }
  }

  corrTemp[ID] = corr;
}

/*
  Compute vorticity
*/
__kernel void cld_computeVorticity(//Input
                                   const __global float4 *predPos,      // 0
                                   const __global uint2  *startEndCell, // 1
                                   const __global float4 *vel,          // 2
                                   //Param
                                   const     FluidParams fluid,         // 3
                                   //Output
                                         __global float4 *vorticity)    // 4
{
  const float4 pos = predPos[ID];
  const float4 velocity = vel[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float4 vort = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          vort += cross((vel[e] - velocity), gradSpiky(pos - predPos[e] - absWallXYZ * signAbsWall, EFFECT_RADIUS));
        }
      }
    }
  }

  vorticity[ID] = vort;
}

/*
  Apply vorticity confinement
*/
__kernel void cld_applyVorticityConfinement(//Input
                                            const __global float4 *predPos,      // 0
                                            const __global uint2  *startEndCell, // 1
                                            const __global float4 *vort,         // 2
                                            //Param
                                            const     FluidParams fluid,         // 3
                                            //Output
                                                  __global float4 *vel)          // 4
{
  const float4 pos = predPos[ID];
  const float4 vorticity = vort[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  // vorticity confinement
  float4 n = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          n += fast_length(vort[e]) * gradSpiky(pos - predPos[e] - absWallXYZ * signAbsWall, EFFECT_RADIUS);
        }
      }
    }
  }

  // Adding vorticity confinement to attenue virtual damping
  vel[ID] += fluid.vorticityConfCoeff * cross(normalize(n), vorticity) * fluid.timeStep;
}


/*
  Apply xsph viscosity correction
*/
__kernel void cld_applyXsphViscosityCorrection(//Input
                                               const __global float4 *predPos,      // 0
                                               const __global uint2  *startEndCell, // 1
                                               const __global float4 *velIn,        // 2
                                               //Param
                                               const     FluidParams fluid,         // 3
                                               //Output
                                                     __global float4 *velOut)       // 4
{
  const float4 pos = predPos[ID];
  const float4 velocity = velIn[ID];
  const int3 cellIndex3D = convert_int3(getCell3DIndexFromPos(pos));

  float4 viscosity = (float4)(0.0f);

  uint cellNIndex1D = 0;
  int3 cellNIndex3D = (int3)(0);
  int3 gridResXYZ = (int3)(GRID_RES_X, GRID_RES_Y, GRID_RES_Z);
  uint2 startEndN = (uint2)(0, 0);

  float4 absWallXYZ = (float4)(ABS_WALL_X, ABS_WALL_Y, ABS_WALL_Z, 0.0f);
  float4 signAbsWall = (float4)(0.0f);

  // 27 cells to visit, current one + 3D neighbors
  for (int iX = -1; iX <= 1; ++iX)
  {
    for (int iY = -1; iY <= 1; ++iY)
    {
      for (int iZ = -1; iZ <= 1; ++iZ)
      {
        signAbsWall = (float4)(0.0f);

        cellNIndex3D = (cellIndex3D + (int3)(iX, iY, iZ) + gridResXYZ) % gridResXYZ;

        // Periodic BC for x, periodic neighbors must be considered
        if((cellIndex3D.x + iX) >= GRID_RES_X) signAbsWall.x = 2.0f;
        else if((cellIndex3D.x + iX) < 0) signAbsWall.x = -2.0f;

        // Wall BC for y, out of domain cells are discarded
        if(((cellIndex3D.y + iY) >= GRID_RES_Y) || ((cellIndex3D.y + iY) < 0)) continue;
        
        // Periodic BC for z, periodic neighbors must be considered
        if((cellIndex3D.z + iZ) >= GRID_RES_Z) signAbsWall.z = 2.0f;
        else if((cellIndex3D.z + iZ) < 0) signAbsWall.z = -2.0f;

        cellNIndex1D = (cellNIndex3D.x * GRID_RES_Y + cellNIndex3D.y) * GRID_RES_Z + cellNIndex3D.z;

        startEndN = startEndCell[cellNIndex1D];

        for (uint e = startEndN.x; e <= startEndN.y; ++e)
        {
          viscosity += (velIn[e] - velocity) * poly6(pos - predPos[e] - absWallXYZ * signAbsWall, EFFECT_RADIUS);
        }
      }
    }
  }

  // Adding xsph viscosity for a more coherent motion
  velOut[ID] = velocity + fluid.xsphViscosityCoeff * viscosity;
}

/*
  Correction position using Constraint correction value
*/
__kernel void cld_correctPosition(//Input
                                  const __global float4 *corrPos, // 0
                                  //Output
                                        __global float4 *predPos) // 1
{
  predPos[ID] += corrPos[ID];
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

/*
  Update position using predicted one and velocity field
*/
__kernel void cld_updatePosition(//Input
                                 const  __global float4 *predPos, // 0
                                 //Param
                                 const     CloudParams cloud,     // 1
                                 //Output
                                        __global float4 *pos)     // 2
{
  pos[ID] = predPos[ID];
  pos[ID].x += (1 - exp(- (predPos[ID].y + ABS_WALL_Y) * 0.2f)) * cloud.windCoeff * cloud.timeStep * (float)(cloud.dim - 2); //0.02f;
  pos[ID].z += (1 - exp(- (predPos[ID].y + ABS_WALL_Y) * 0.3f)) * 0.7f * cloud.windCoeff * cloud.timeStep; //0.015f;
}

/*
  Update velocity buffer
*/
__kernel void cld_updateVel(//Input
                            const __global float4 *totCorrPos,    // 0
                            //Param
                            const     FluidParams fluid,       // 1
                            //Output
                                  __global float4 *vel)        // 2
{
  // Clamping velocity and preventing division by 0
  vel[ID] = clamp((totCorrPos[ID]) / (fluid.timeStep + FLOAT_EPS), -MAX_VEL, MAX_VEL);
}