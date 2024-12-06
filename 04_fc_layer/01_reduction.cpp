void fully_connected_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
	int input_dim, int output_dim, int batch_size)
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
	err = clSetKernelArg(kernel, 5, sizeof(int), &output_dim);
	CHECK_ERROR(err);

	size_t global_work_size[2] = { output_dim*64, batch_size };
	size_t local_size[2] = { 64,1 };
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
}
