void fusion_layer(cl_command_queue queue, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, cl_mem* weights, cl_mem* biases,
    int input_dim, int output_dim, int nbyn, int batch_size)
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

    size_t global_work_size[2] = { (nbyn / 2) * (nbyn / 2) * output_dim, batch_size };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}
