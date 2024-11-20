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

void cnn_init(cl_context *context, cl_command_queue *queue, cl_program *program) {
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
    int input_dim, int output_dim, int nbyn)
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

	
    
	if (input_dim == 3) { // 처음에 입력 채널이 3일 때는 워크 사이즈가 나누어 떨어지지 않으므로 그냥 일반 컨볼루션 커널 사용
		size_t global_work_size[2] = { nbyn * nbyn, output_dim };
		err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
		CHECK_ERROR(err);
	} 
	else {
		// 워크 사이즈를 3차원으로 입력해야 locality가 맞아서 실행되는 것 같음.
		size_t global_work_size[3] = { nbyn, nbyn, output_dim };
		size_t local_work_size[3] = { 2, 2, 32 }; // 임의로 설정함. 가로 세로 크기는 2가 적당해 보임, 채널은 실험 필요
		err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		CHECK_ERROR(err);
	}
}

// 그냥 컨볼루션 레이어
//void convolution_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
//	int input_dim, int output_dim, int nbyn)
//{
//	cl_int err;
//	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
//	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
//	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
//	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
//	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel, 4, sizeof(int), &input_dim);
//	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel, 5, sizeof(int), &nbyn);
//	CHECK_ERROR(err);
//
//	size_t global_work_size[2] = { nbyn * nbyn, output_dim };
//	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
//	CHECK_ERROR(err);
//}

void max_pooling_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs,
    int dim, int nbyn)
{
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(int), &nbyn);
	CHECK_ERROR(err);

    size_t global_work_size[2] = { nbyn * nbyn, dim };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void fully_connected_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim)
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

    size_t global_work_size = output_dim;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
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

	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cnn_init(&context, &queue, &program);

	cl_kernel conv_kernel = clCreateKernel(program, "conv_kernel", &err);
	CHECK_ERROR(err);
	cl_kernel conv_lm_kernel = clCreateKernel(program, "conv_lm_kernel", &err);
	CHECK_ERROR(err);
	cl_kernel pooling_kernel = clCreateKernel(program, "pooling_kernel", &err);
	CHECK_ERROR(err);
	cl_kernel fc_kernel = clCreateKernel(program, "fc_kernel", &err);
	CHECK_ERROR(err);


	time_t start, end;
	start = clock();

	cl_mem wBuf[21];
	cl_mem bBuf[21];

	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
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
		layerBuf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i], NULL, &err);
		CHECK_ERROR(err);
	}
	float* outputLayer = (float*)malloc(sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20]);
	if (outputLayer == NULL) 
		perror("malloc error");

	cl_mem imageBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 32 * 32, NULL, &err);
	CHECK_ERROR(err);

	int mode = 1;
	// run network
	if (mode == 1) { // 로컬 메모리를 사용하는 컨볼루션 레이어 사용
		for (int i = 0; i < num_of_image; ++i) {
			err = clEnqueueWriteBuffer(queue, imageBuf, CL_TRUE, 0, sizeof(float) * 3 * 32 * 32, images, 0, NULL, NULL);
			CHECK_ERROR(err);

			convolution_layer(queue, conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[1], &layerBuf[2], INPUT_DIM[2], NBYN[2]);

			convolution_layer(queue, conv_lm_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[4], &layerBuf[5], INPUT_DIM[5], NBYN[5]);

			convolution_layer(queue, conv_lm_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[8], &layerBuf[9], INPUT_DIM[9], NBYN[9]);

			convolution_layer(queue, conv_lm_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[12], &layerBuf[13], INPUT_DIM[13], NBYN[13]);

			convolution_layer(queue, conv_lm_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
			convolution_layer(queue, conv_lm_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[16], &layerBuf[17], INPUT_DIM[17], NBYN[17]);

			fully_connected_layer(queue, fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18], INPUT_DIM[18], OUTPUT_DIM[18]);
			fully_connected_layer(queue, fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19], INPUT_DIM[19], OUTPUT_DIM[19]);
			fully_connected_layer(queue, fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20], INPUT_DIM[20], OUTPUT_DIM[20]);

			err = clEnqueueReadBuffer(queue, layerBuf[20], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20], outputLayer, 0, NULL, NULL);
			CHECK_ERROR(err);

			softmax(outputLayer, 10);

			labels[i] = find_max(outputLayer, 10);
			confidences[i] = outputLayer[labels[i]];
			images += 32 * 32 * 3;
		}
	}
	if (mode == 2) { // 그냥 컨볼루션 레이어 사용
		for (int i = 0; i < num_of_image; ++i) {
			err = clEnqueueWriteBuffer(queue, imageBuf, CL_TRUE, 0, sizeof(float) * 3 * 32 * 32, images, 0, NULL, NULL);
			CHECK_ERROR(err);

			convolution_layer(queue, conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
			convolution_layer(queue, conv_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[1], &layerBuf[2], INPUT_DIM[2], NBYN[2]);

			convolution_layer(queue, conv_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
			convolution_layer(queue, conv_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[4], &layerBuf[5], INPUT_DIM[5], NBYN[5]);

			convolution_layer(queue, conv_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
			convolution_layer(queue, conv_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
			convolution_layer(queue, conv_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[8], &layerBuf[9], INPUT_DIM[9], NBYN[9]);

			convolution_layer(queue, conv_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
			convolution_layer(queue, conv_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
			convolution_layer(queue, conv_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[12], &layerBuf[13], INPUT_DIM[13], NBYN[13]);

			convolution_layer(queue, conv_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
			convolution_layer(queue, conv_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
			convolution_layer(queue, conv_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
			max_pooling_layer(queue, pooling_kernel, &layerBuf[16], &layerBuf[17], INPUT_DIM[17], NBYN[17]);

			fully_connected_layer(queue, fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18], INPUT_DIM[18], OUTPUT_DIM[18]);
			fully_connected_layer(queue, fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19], INPUT_DIM[19], OUTPUT_DIM[19]);
			fully_connected_layer(queue, fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20], INPUT_DIM[20], OUTPUT_DIM[20]);

			err = clEnqueueReadBuffer(queue, layerBuf[20], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20], outputLayer, 0, NULL, NULL);
			CHECK_ERROR(err);

			softmax(outputLayer, 10);

			labels[i] = find_max(outputLayer, 10);
			confidences[i] = outputLayer[labels[i]];
			images += 32 * 32 * 3;
		}
	}
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 21; ++i) {
		err = clReleaseMemObject(layerBuf[i]);
		CHECK_ERROR(err);
	}
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
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
	err = clReleaseKernel(conv_lm_kernel);
	CHECK_ERROR(err);
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


