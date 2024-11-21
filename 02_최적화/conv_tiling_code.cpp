void convolution_layer_tiling(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
	int inDim, int outDim, int nbyn)
{
	int TILE_SIZE = 8;
	cl_int err;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), weights);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), biases);
	clSetKernelArg(kernel, 4, sizeof(int), &inDim);
	clSetKernelArg(kernel, 5, sizeof(int), &outDim);
	clSetKernelArg(kernel, 6, sizeof(int), &nbyn);
	clSetKernelArg(kernel, 7, sizeof(int), &TILE_SIZE);

	// 글로벌 및 로컬 워크그룹 크기 설정
	size_t global_work_size[2] = { nbyn, nbyn };
	size_t local_work_size[2] = { TILE_SIZE, TILE_SIZE };

	// OpenCL 커널 실행
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clFinish(queue);
}