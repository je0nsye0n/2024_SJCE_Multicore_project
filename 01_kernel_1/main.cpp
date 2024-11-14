#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include <CL/cl.h>

#define _CRT_SECURE_NO_WARNINGS

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
size_t kernel_source_size;
char* kernel_source;
float* layer[21];

char* get_source_code(const char* file_name, size_t * len) {
	FILE* file;
	errno_t err = fopen_s(&file, file_name, "rb");
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


extern const char* CLASS_NAME[];

void cnn_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);
}

static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {

	memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
	int x = 0, y = 0;
	int offset = nbyn * nbyn;
	float sum = 0, temp;
	float* input, * output;

	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		input = inputs;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			output = outputs;
			for (int row = 0; row < nbyn; ++row) {
				for (int col = 0; col < nbyn; ++col) {
					sum = 0;
					for (int fRow = 0; fRow < 3; ++fRow) {
						for (int fCol = 0; fCol < 3; ++fCol) {
							x = col + fCol - 1;
							y = row + fRow - 1;

							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
								sum += input[nbyn * y + x] * filter[3 * fRow + fCol];
							}

						}
					}
					*(output++) += sum;
				}
			}
			filter += 9;
			input += offset;

		}
		for (int i = 0; i < offset; ++i) {
			(*outputs) = (*outputs) + (*biases);
			if (*outputs < 0) (*outputs) = 0;	//ReLU
			outputs++;
		}
		++biases;
	}

}

static void max_pooling(float* input, float* output, int DIM, int nbyn) {
	float max, temp;
	int n, row, col, x, y;
	for (n = 0; n < DIM; ++n) {
		for (row = 0; row < nbyn; row += 2) {
			for (col = 0; col < nbyn; col += 2) {
				//max = -FLT_MAX;
				max = 0;
				for (y = 0; y < 2; ++y) {
					for (x = 0; x < 2; ++x) {
						temp = input[nbyn * (row + y) + col + x];
						if (max < temp) max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}
}


void max_pooling_gpu(float* input, float* output, int DIM, int nbyn) {
	// DIM * nbyn * nbyn 입력 크기 및 DIM * (nbyn / 2) * (nbyn / 2) 출력 크기
	cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DIM * nbyn * nbyn, NULL, &err);
	cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DIM * (nbyn / 2) * (nbyn / 2), NULL, &err);
	CHECK_ERROR(err);

	// 입력 데이터를 입력 버퍼에 복사
	err = clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, sizeof(float) * DIM * nbyn * nbyn, input, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 커널 인자 설정
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &nbyn);
	CHECK_ERROR(err);

	// 글로벌 워크 크기 설정 (각 채널의 2x2 풀링 결과를 병렬로 처리)
	size_t global_size[3] = { DIM, nbyn, nbyn };
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 출력 버퍼에서 결과를 읽어옴 (출력 크기는 DIM * (nbyn / 2) * (nbyn / 2))
	err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * DIM * (nbyn / 2) * (nbyn / 2), output, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 메모리 해제
	clReleaseMemObject(input_buf);
	clReleaseMemObject(output_buf);
}


void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
	}
}

void fc_layer_gpu(float* input, float* output, float* weights, float* biases, int inDim, int outDim, int i) {

	// 버퍼 생성
	cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * inDim, NULL, &err);
	cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outDim * sizeof(float), NULL, &err);
	cl_mem w_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, inDim * outDim * sizeof(float), NULL, &err);
	cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, outDim * sizeof(float), NULL, &err);
	CHECK_ERROR(err);

	// 버퍼에 데이터 전송
	err = clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, sizeof(float) * inDim, input, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, w_buf, CL_TRUE, 0, inDim * outDim * sizeof(float), weights, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, b_buf, CL_TRUE, 0, outDim * sizeof(float), biases, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 커널 인자 설정
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &w_buf);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_buf);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &inDim);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &outDim);
	CHECK_ERROR(err);

	// 워크그룹 크기 설정
	size_t global_size = outDim;

	// 커널 실행
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 결과를 output 배열로 읽기
	err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, outDim * sizeof(float), output, 0, NULL, NULL);
	CHECK_ERROR(err);

	// 메모리 해제
	clReleaseMemObject(input_buf);
	clReleaseMemObject(output_buf);
	clReleaseMemObject(w_buf);
	clReleaseMemObject(b_buf);
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

	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}

	cnn_init();
	// 커널 소스 읽기
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// 커널 빌드
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("Build log:\n%s\n", log);
		free(log);
		CHECK_ERROR(err);
	}

	kernel = clCreateKernel(program, "fc_layer_kernel", &err);
	kernel = clCreateKernel(program, "max_pooling_kernel", &err);
	CHECK_ERROR(err);

	time_t start, end;
	start = clock();
	// run network
	for (int i = 0; i < num_of_image; ++i) {
		convolution(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling_gpu(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling_gpu(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling_gpu(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling_gpu(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

		softmax(layer[20], 10);

		labels[i] = find_max(layer[20], 10);
		confidences[i] = layer[20][labels[i]];
		images += 32 * 32 * 3;
	}
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(kernel_source);

}
