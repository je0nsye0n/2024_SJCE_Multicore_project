#pragma warning(disable : 4996)
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include <CL/cl.h>

extern const char* CLASS_NAME[];

cl_context context;
cl_command_queue queue;
cl_command_queue read_queue, cnn_queue, save_queue;
cl_command_queue cnn_queue_list[6];

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void cnn_init(cl_context* context, cl_command_queue* queue, cl_command_queue* save_queue, cl_command_queue* read_queue, cl_command_queue* cnn_queue, cl_command_queue* cnn_queue_list,cl_program* program ) {
    cl_int err;

    // Platform ID
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    *context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create Command Queue
    *queue = clCreateCommandQueueWithProperties(*context, device, 0, &err);
    *read_queue = clCreateCommandQueueWithProperties(*context, device, 0, &err);
    *cnn_queue = clCreateCommandQueueWithProperties(*context, device, 0, &err);
    *save_queue = clCreateCommandQueueWithProperties(*context, device, 0, &err);
    for (int i = 0; i < 6; i++) {
        *(cnn_queue_list+i) = clCreateCommandQueueWithProperties(*context, device, 0, &err);
    }
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    *program = clCreateProgramWithSource(*context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(*program, 1, &device, "", NULL, NULL);
    build_error(*program, device, err);
    CHECK_ERROR(err);

    free(kernel_source);
}


// 로컬 메모리를 사용하는 컨볼루션 레이어
void convolution_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim, int nbyn, int batch_size, cl_event* before, cl_event* after)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &batch_size);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &output_dim);
    CHECK_ERROR(err);

    int cnt = 1;
    if (before == NULL) cnt = 0;

    size_t global_work_size[2] = { nbyn * nbyn * output_dim, batch_size };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, cnt, before, after);
    CHECK_ERROR(err);
}

void winograd_convolution_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim, int nbyn, int batch_size, cl_event* before, cl_event* after)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &batch_size);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &output_dim);
    CHECK_ERROR(err);

    // 워크 그룹 크기를 최적화하여 GPU의 병렬성 극대화
    size_t local_work_size[2] = { 16, 16 }; // 로컬 워크 사이즈 설정
    size_t global_work_size[2] = { (size_t)nbyn, (size_t)nbyn * batch_size };

    int cnt = 1;
    if (before == NULL) cnt = 0;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, cnt, NULL, NULL);
    CHECK_ERROR(err);
}

void max_pooling_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs,
    int dim, int nbyn, int batch_size, cl_event* before, cl_event* after)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &batch_size);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &dim);
    CHECK_ERROR(err);

    int cnt = 1;
    if (before == NULL) cnt = 0;
    size_t global_work_size[2] = { nbyn * nbyn * dim, batch_size };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, cnt, before, after);
    CHECK_ERROR(err);
}

void fused_conv_pool_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim, int nbyn, int batch_size, cl_event* before, cl_event* after)
{
    cl_int err;
    // Setting kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(float) * nbyn * nbyn, NULL); // 로컬 메모리 할당 크기 조정
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &batch_size);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 8, sizeof(int), &output_dim);
    CHECK_ERROR(err);

    // Define the global and local work size
    size_t global_work_size[2] = { nbyn * nbyn * output_dim, batch_size };
    size_t local_work_size[2] = { 4, 4 }; // 로컬 워크 사이즈 축소
    // Enqueue the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}


void fully_connected_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim, int batch_size, cl_event* before, cl_event* after)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &batch_size);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &output_dim);
    CHECK_ERROR(err);

    size_t global_work_size[2] = { output_dim, batch_size };
    int cnt = 1;
    if (before == NULL) cnt = 0;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, cnt, before, after);
    CHECK_ERROR(err);
}

void save_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, int* labels, float* confidences, int batch_size,
    cl_event* before, cl_event* after) {
    cl_int err;

    cl_mem labels_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * batch_size, NULL, &err);
    CHECK_ERROR(err);

    cl_mem confidences_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * batch_size, NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);  // inputs를 GPU 메모리로 변환해야 함
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &labels_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &confidences_buf);
    CHECK_ERROR(err);
    
    size_t global_work_size = batch_size;
    int cnt = 1;
    if (before == NULL) cnt = 0;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, cnt, before, after);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, labels_buf, CL_FALSE, 0, sizeof(int) * batch_size, labels, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, confidences_buf, CL_FALSE, 0, sizeof(float) * batch_size, confidences, 0, NULL, NULL);
    CHECK_ERROR(err);

    clReleaseMemObject(labels_buf);
    clReleaseMemObject(confidences_buf);

    //for (int i = 0; i < batch_size; i++) {
     //   printf("%d: %d %lf\n", i, *(labels + i), *(confidences + i));
    //}

}


static void softmax(float* input, int N) {
    int i;
    float max = input[0];
    for (i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    for (i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7);
    }
}

static int find_max(float* input, int classNum) {
    int i;
    int maxIndex = 0;
    float max = 0;
    for (i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}



const int INPUT_DIM[] = {
    3, 64,
    64,

    64,128,
    128,

    128, 256, 256,
    256,

    256, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    512
};

const int OUTPUT_DIM[] = {
    64, 64,
    64,

    128, 128,
    128,

    256, 256, 256,
    256,

    512, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    10
};

const int NBYN[] = {
    32, 32,
    16,

    16, 16,
    8,

    8, 8, 8,
    4,

    4, 4, 4,
    2,

    2, 2, 2,
    1,

    1,
    1,
    1
};



void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
    cl_int err;

    cl_program program;
    cnn_init(&context, &queue, &save_queue,&read_queue,&cnn_queue,cnn_queue_list,&program);

    cl_kernel conv_kernel = clCreateKernel(program, "conv_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel pooling_kernel = clCreateKernel(program, "pooling_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel fc_kernel = clCreateKernel(program, "fc_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel winograd_conv_kernel = clCreateKernel(program, "winograd_conv_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel save_kernel = clCreateKernel(program, "save_kernel", &err);
    CHECK_ERROR(err);


    time_t start, end;

    cl_mem wBuf[21];
    cl_mem bBuf[21];

    int offset = 0;

    int batch_size = 64;

    // link weights and biases to network
    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;    // pooling layer has no weights and biases
        wBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * INPUT_DIM[i] * 3 * 3, NULL, &err);
        CHECK_ERROR(err);
        bBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, wBuf[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i] * INPUT_DIM[i] * 3 * 3, network + offset, 0, NULL, NULL);
        CHECK_ERROR(err);
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
        err = clEnqueueWriteBuffer(queue, bBuf[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i], network + offset, 0, NULL, NULL);
        CHECK_ERROR(err);
        offset += OUTPUT_DIM[i];
    }
    for (int i = 18; i < 21; ++i) {
        wBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * INPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
        bBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, wBuf[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i] * INPUT_DIM[i], network + offset, 0, NULL, NULL);
        CHECK_ERROR(err);
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        err = clEnqueueWriteBuffer(queue, bBuf[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i], network + offset, 0, NULL, NULL);
        CHECK_ERROR(err);
        offset += OUTPUT_DIM[i];
    }

    cl_mem layerBuf[21];
    for (int i = 0; i < 21; ++i) {
        layerBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, batch_size * sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i], NULL, &err);
        CHECK_ERROR(err);
    }
    float* outputLayer = (float*)malloc(sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20] * batch_size);
    if (outputLayer == NULL)
        perror("malloc error");

    cl_mem imageBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 32 * 32 * batch_size, NULL, &err);
    CHECK_ERROR(err);

    cl_event write_event, conv_event, pool_event[5], read_event, save_event;

    // run network
    start = clock();
    for (int i = 0; i < num_of_image; i += batch_size) {
        int current_batch_size = (i + batch_size > num_of_image) ? (num_of_image - i) : batch_size;

        if (i == 0) {

            err = clEnqueueWriteBuffer(read_queue, imageBuf, CL_FALSE, 0,
                sizeof(float) * 3 * 32 * 32 * current_batch_size,
                images + i * 32 * 32 * 3, 0, NULL, &write_event);
            CHECK_ERROR(err);

            convolution_layer(cnn_queue_list[0], conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0],
                INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], current_batch_size,
                &write_event, NULL);

            convolution_layer(cnn_queue_list[0], conv_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1],
                INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[0], pooling_kernel, &layerBuf[1], &layerBuf[2],
                INPUT_DIM[2], NBYN[2], current_batch_size,
                NULL, &pool_event[0]);



            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3],
                INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], current_batch_size,
                &pool_event[0], NULL);

            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4],
                INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[1], pooling_kernel, &layerBuf[4], &layerBuf[5],
                INPUT_DIM[5], NBYN[5], current_batch_size,
                NULL, &pool_event[1]);



            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6],
                INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], current_batch_size,
                &pool_event[1], NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7],
                INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8],
                INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[2], pooling_kernel, &layerBuf[8], &layerBuf[9],
                INPUT_DIM[9], NBYN[9], current_batch_size,
                NULL, &pool_event[2]);



            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10],
                INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], current_batch_size,
                &pool_event[2], NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11],
                INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12],
                INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[3], pooling_kernel, &layerBuf[12], &layerBuf[13],
                INPUT_DIM[13], NBYN[13], current_batch_size,
                NULL, &pool_event[3]);



            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14],
                INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], current_batch_size,
                &pool_event[3],NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15],
                INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16],
                INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[4], pooling_kernel, &layerBuf[16], &layerBuf[17],
                INPUT_DIM[17], NBYN[17], current_batch_size,
                NULL, &pool_event[4]);


            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18],
                INPUT_DIM[18], OUTPUT_DIM[18], current_batch_size,
                &pool_event[4], NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19],
                INPUT_DIM[19], OUTPUT_DIM[19], current_batch_size,
                NULL, NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20],
                INPUT_DIM[20], OUTPUT_DIM[20], current_batch_size,
                NULL, &conv_event);


            //err = clEnqueueReadBuffer(queue, layerBuf[20], CL_FALSE, 0,
                //sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20] * current_batch_size,
                //outputLayer, 1, &conv_event[15], &read_event);
            //CHECK_ERROR(err);

            save_layer(save_queue, save_kernel, &layerBuf[20], &labels[i], &confidences[i], current_batch_size, &conv_event, &save_event);

        }
        else {

            err = clEnqueueWriteBuffer(read_queue, imageBuf, CL_FALSE, 0,
                sizeof(float) * 3 * 32 * 32 * current_batch_size,
                images + i * 32 * 32 * 3, 1, &pool_event[0], &write_event);
            CHECK_ERROR(err);
            clReleaseEvent(pool_event[0]);

            convolution_layer(cnn_queue_list[0], conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0],
                INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], current_batch_size,
                &pool_event[1], NULL);
            clReleaseEvent(pool_event[0]);

            convolution_layer(cnn_queue_list[0], conv_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1],
                INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[0], pooling_kernel, &layerBuf[1], &layerBuf[2],
                INPUT_DIM[2], NBYN[2], current_batch_size,
                NULL, &pool_event[0]);



            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3],
                INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], current_batch_size,
                &pool_event[1], NULL);
            clReleaseEvent(pool_event[1]);

            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4],
                INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[1], pooling_kernel, &layerBuf[4], &layerBuf[5],
                INPUT_DIM[5], NBYN[5], current_batch_size,
                NULL, &pool_event[1]);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6],
                INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], current_batch_size,
                &pool_event[2],NULL);
            clReleaseEvent(pool_event[2]);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7],
                INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8],
                INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[2], pooling_kernel, &layerBuf[8], &layerBuf[9],
                INPUT_DIM[9], NBYN[9], current_batch_size,
                NULL, &pool_event[2]);


            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10],
                INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], current_batch_size,
                &pool_event[3], NULL);
            clReleaseEvent(pool_event[3]);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11],
                INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12],
                INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[3], pooling_kernel, &layerBuf[12], &layerBuf[13],
                INPUT_DIM[13], NBYN[13], current_batch_size,
                NULL, &pool_event[3]);


            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14],
                INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], current_batch_size,
                &pool_event[4], NULL);
            clReleaseEvent(pool_event[4]);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15],
                INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16],
                INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[4], pooling_kernel, &layerBuf[16], &layerBuf[17],
                INPUT_DIM[17], NBYN[17], current_batch_size,
                NULL, &pool_event[4]);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18],
                INPUT_DIM[18], OUTPUT_DIM[18], current_batch_size,
                &conv_event, NULL);
            clReleaseEvent(conv_event);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19],
                INPUT_DIM[19], OUTPUT_DIM[19], current_batch_size,
                NULL, NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20],
                INPUT_DIM[20], OUTPUT_DIM[20], current_batch_size,
                NULL, &conv_event);

            //err = clEnqueueReadBuffer(queue, layerBuf[20], CL_FALSE, 0,
            //    sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20] * current_batch_size,
            //    outputLayer, 1, &conv_event[15], &read_event);
            //CHECK_ERROR(err);

            save_layer(save_queue, save_kernel, &layerBuf[20], &labels[i], &confidences[i], current_batch_size, &conv_event, &save_event);
        }


        //for (int b = 0; b < current_batch_size; ++b) {
            //softmax(outputLayer + b * 10, 10);    
            //labels[i + b] = find_max(outputLayer + b * 10, 10);
            //confidences[i + b] = outputLayer[b * 10 + labels[i + b]];
            // => 커널 변경?
        //}

        //clWaitForEvents(1, &output_event);
        //clReleaseEvent(conv_event[1]);

    }
    end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

    for (int i = 0; i < 21; ++i) {
        err = clReleaseMemObject(layerBuf[i]);
        CHECK_ERROR(err);
    }

    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;    // pooling layer has no weights and biases
        err = clReleaseMemObject(wBuf[i]);
        CHECK_ERROR(err);
        err = clReleaseMemObject(bBuf[i]);
        CHECK_ERROR(err);
    }

    for (int i = 18; i < 21; ++i) {
        err = clReleaseMemObject(wBuf[i]);
        CHECK_ERROR(err);
        err = clReleaseMemObject(bBuf[i]);
        CHECK_ERROR(err);
    }

    err = clReleaseKernel(pooling_kernel);
    CHECK_ERROR(err);
    err = clReleaseKernel(fc_kernel);
    CHECK_ERROR(err);
    err = clReleaseProgram(program);
    CHECK_ERROR(err);
    err = clReleaseCommandQueue(queue);
    CHECK_ERROR(err);
    err = clReleaseContext(context);
    CHECK_ERROR(err);

    free(outputLayer);
}
