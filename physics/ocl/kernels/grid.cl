// Preprocessor defines following constant variables in Boids.cpp
// ABS_WALL_POS            - absolute position of the walls in x,y,z
// GRID_RES                - resolution of the grid
// GRID_CELL_SIZE          - size of a cell, size / res of grid
// GRID_NUM_CELLS          - total number of cells in the grid
// NUM_MAX_PARTS_IN_CELL   - maximum number of particles taking into account in a single cell in simplified mode
#define FLOAT_EPSILON 0.01f
#define ID            get_global_id(0)

/*
  Compute 3D index of the cell containing given position
*/
inline uint3 getCell3DIndexFromPos(float4 pos)
{
  // Moving particles in [0 - 2 * ABS_WALL_POS] to have coords matching with cellIndices
  const float3 posXYZ = clamp(pos.xyz, -ABS_WALL_POS, ABS_WALL_POS) + (float3)(ABS_WALL_POS);

  const uint3 cell3DIndex = convert_uint3(floor(posXYZ / GRID_CELL_SIZE));

  return cell3DIndex;
}

/*
  Compute 1D index of the cell containing given position
*/
inline uint getCell1DIndexFromPos(float4 pos)
{
  const uint3 cell3DIndex = getCell3DIndexFromPos(pos);

  const uint cell1DIndex = cell3DIndex.x * GRID_RES * GRID_RES
                         + cell3DIndex.y * GRID_RES
                         + cell3DIndex.z;

  return cell1DIndex;
}

/*
  Reset grid detector buffer. For rendering purpose only.
*/
__kernel void resetGridDetector(__global float8* gridDetector)
{
  gridDetector[ID] = (float8)(0.0f);
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
  Reset startEndPartID buffer for each cell.
*/
__kernel void resetStartEndCell(__global uint2 *cStartEndPartID)
{
  // Resetting with 1 as starting index and 0 as ending index
  // Little hack to bypass empty cell further
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
