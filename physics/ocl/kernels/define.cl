
// Commonly used defined variables
// define.cl must be set as the first .cl file to create the OpenCL program
#define ID            get_global_id(0)

#define FLOAT_EPS     0.00000001f

#define GRAVITY_ACC_Y -9.81f
#define GRAVITY_ACC   (float4)(0.0f, GRAVITY_ACC_Y, 0.0f, 0.0f)
#define FAR_DIST      100000000.0f
