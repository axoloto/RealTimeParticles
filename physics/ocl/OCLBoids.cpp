#include "OCLBoids.hpp"
#include "CL/cl_gl.h"  // WIP
#include "windows.h"  // WIP

using namespace Core;

#define PROGRAM_FILE "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\matvec.cl"

#define KERNEL_RANDOM_FUNC "randPosVerts"
#define KERNEL_COLOR_FUNC "colorVerts"

static void OCLExample();
static bool isOCLExtensionSupported(cl_device_id device, const char* extension);

OCLBoids::OCLBoids(int boxSize, int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO) : Boids(boxSize, numEntities)
{
    m_init = initOpenCL();

    if(m_init)
    {
        createRandomKernel(pointCloudCoordVBO);
        createColorKernel(pointCloudColorVBO);
        runKernel(cl_colorKernel, cl_colorBuff);
    }
}

OCLBoids::~OCLBoids()
{
    clReleaseKernel(cl_colorKernel);
    clReleaseMemObject(cl_colorBuff);

    clReleaseKernel(cl_randomKernel);
    clReleaseMemObject(cl_randomBuff);

    clReleaseCommandQueue(cl_queue);
    clReleaseProgram(cl_program);
    clReleaseContext(cl_context);
}

void OCLBoids::updatePhysics()
{
    runKernel(cl_randomKernel, cl_randomBuff);
}

bool OCLBoids::initOpenCL()
{
    cl_int err;

    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    
    clGetPlatformIDs(1, &cl_platform, NULL);
    clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, NULL);

    if(!isOCLExtensionSupported(cl_device, "cl_khr_gl_sharing"))
    {
        printf("error, extension missing to do inter operation between opencl and opengl");
        return false;
    }

    cl_context_properties props[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties) cl_platform,
        0
    };

    cl_context = clCreateContext(props, 1, &cl_device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("error when creating context");
        return false;
    }

    program_handle = fopen(PROGRAM_FILE, "rb");
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';

    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    cl_program = clCreateProgramWithSource(cl_context, 1, (const char **)&program_buffer, &program_size, &err);
    if (err != CL_SUCCESS)
    {
        printf("error when creating program");
        return false;
    }
    free(program_buffer);
    
    err = clBuildProgram(cl_program, 1, &cl_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        clGetProgramBuildInfo(cl_program, cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(cl_program, cl_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return false;
    }

    cl_queue = clCreateCommandQueue(cl_context, cl_device, 0, &err);
    if (err != CL_SUCCESS)
    {
        printf("error when creating queue");
        return false;
    }

    return true;
}

bool OCLBoids::createColorKernel(unsigned int bufferGL)
{
    cl_int err;

    cl_colorKernel = clCreateKernel(cl_program, KERNEL_COLOR_FUNC, &err);
    if (err != CL_SUCCESS) printf("error when creating color kernel");

    cl_colorBuff = clCreateFromGLBuffer(cl_context, CL_MEM_WRITE_ONLY, bufferGL, &err);
    if (err != CL_SUCCESS) printf("error when creating color buffer");

    clSetKernelArg(cl_colorKernel, 0, sizeof(cl_mem), &cl_colorBuff);

    return true;
}

bool OCLBoids::createRandomKernel(unsigned int bufferGL)
{
    cl_int err;

    cl_randomKernel = clCreateKernel(cl_program, KERNEL_RANDOM_FUNC, &err);
    if (err != CL_SUCCESS) printf("error when creating random kernel");

    cl_randomBuff = clCreateFromGLBuffer(cl_context, CL_MEM_WRITE_ONLY, bufferGL, &err);
    if (err != CL_SUCCESS) printf("error when creating random buffer");

    clSetKernelArg(cl_randomKernel, 0, sizeof(cl_mem), &cl_randomBuff);

    return true;
}

void OCLBoids::runKernel(cl_kernel kernel, cl_mem GLCLbuffer )
{
    if(!m_init) return;

    cl_int err = clEnqueueAcquireGLObjects(cl_queue, 1, &GLCLbuffer, 0, NULL, NULL);
    if (err != CL_SUCCESS) printf("error when acquiring GL buffer");

    size_t numWorkItems = NUM_MAX_ENTITIES;
    clEnqueueNDRangeKernel(cl_queue, kernel, 1, NULL, &numWorkItems, NULL, 0, NULL, NULL);

    err = clEnqueueReleaseGLObjects(cl_queue, 1, &GLCLbuffer, 0, NULL, NULL);
    if (err != CL_SUCCESS) printf("error when releasing GL buffer");

    clFinish(cl_queue);
}

static bool isOCLExtensionSupported(cl_device_id device, const char* extension)
{
    if(extension == NULL || extension[0] == '\0') return false;

    char * where = (char *) strchr(extension, ' ');
    if(where != NULL) return false;

    size_t extensionSize;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);

    char *extensions = new char [extensionSize];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);
    
    bool foundExtension = false;
    for(char* start = extensions; ;)
    {
        where = (char *) strstr((const char *) start, extension);
        char* terminator = where + strlen(extension); 

        if(*terminator == ' ' || *terminator == '\0' || *terminator == '\r' || *terminator == '\n')
        {
            foundExtension = true;
            break;
        }

        start = terminator;
    }

    delete[] extensions;

    return foundExtension;
}

static void OCLExample()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, err;
    cl_program program;

    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;
    size_t work_units_per_kernel;
    float mat[16], vec[4], result[4];
    float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cl_mem mat_buff, vec_buff, res_buff;

    for (i = 0; i < 16; i++)
    {
        mat[i] = i * 2.0f;
    }

    for (i = 0; i < 4; i++)
    {
        vec[i] = i * 3.0f;
        correct[0] += mat[i] * vec[i];
        correct[1] += mat[i + 4] * vec[i];
        correct[2] += mat[i + 8] * vec[i];
        correct[3] += mat[i + 12] * vec[i];
    }

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    program_handle = fopen(PROGRAM_FILE, "rb");
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';

    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer, &program_size, &err);
    if (err != CL_SUCCESS)
    {
        printf("error in program");
    }
    free(program_buffer);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    kernel = clCreateKernel(program, KERNEL_RANDOM_FUNC, &err);

    queue = clCreateCommandQueue(context, device, 0, &err);

    mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, mat, &err);
    vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, vec, &err);
    res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 4, NULL, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);

    work_units_per_kernel = 4;

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float) * 4, result, 0, NULL, NULL);

    if ((result[0] == correct[0]) && (result[1] == correct[1]) && (result[2] == correct[2]) && (result[3] == correct[3]))
    {
        printf("Matrix-vector multiplication successful.\n");
    }
    else
    {
        printf("Matrix-vector multiplication unsuccessful.\n");
    }

    clReleaseMemObject(mat_buff);
    clReleaseMemObject(vec_buff);
    clReleaseMemObject(res_buff);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}
