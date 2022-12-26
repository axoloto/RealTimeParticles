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
  float coeff = 1.0f; // to define
  float cte = 1.0f; // to define
  return - coeff * (altitude + ABS_WALL_POS) + cte;
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
  temp[ID] = tempIn[ID] - cloud.adiabaticLapseRate * vel[ID].y;
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
  float A = 100.0f;
  float B = -3;
  float C = 2.3;
  cloudGen[ID] = cloud.phaseTransitionRate * (vaporDens[ID] - min(A * exp(B / (tempIn[ID] + C)), cloudDens[ID] + vaporDens[ID]));
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
                                 //Param
                                 const      FluidParams fluid,    // 1
                                 //Output
                                        __global float4 *col)     // 2
{
  float4 blue      = (float4)(0.0f, 0.1f, 1.0f, 1.0f);
  float4 lightBlue = (float4)(0.7f, 0.7f, 1.0f, 1.0f);
  float4 darkBlue  = (float4)(0.0f, 0.0f, 0.8f, 1.0f);

  float constraint = (1.0f - density[ID] / fluid.restDensity);

  float4 color = blue;

  if(constraint > 0.0f)
    color += constraint * (lightBlue - blue) / 0.35f;
  else if(constraint < 0.0f)
    color += constraint * (blue - darkBlue) / 0.35f;

  col[ID] = color;
}