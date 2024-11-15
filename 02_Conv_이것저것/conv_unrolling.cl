__kernel void conv_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* filter,
	__global float* biases,
	const int inDim,
	const int nbyn
) {
	const int row = get_global_id(0);
	const int col = get_global_id(1);
	const int outNeuron = get_global_id(2);

	const int offset = nbyn * nbyn;
	float sum = 0.0f;

	for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		int inputOffset = inNeuron * offset + row * nbyn + col;
		int filterOffset = (outNeuron * inDim + inNeuron) * 9;

		// 필터의 3x3 요소를 직접 펼쳐서 반복문을 없앰
		sum += (col > 0 && row > 0) ? inputs[inputOffset - nbyn - 1] * filter[filterOffset] : 0.0f;
		sum += (row > 0) ? inputs[inputOffset - nbyn] * filter[filterOffset + 1] : 0.0f;
		sum += (col < nbyn - 1 && row > 0) ? inputs[inputOffset - nbyn + 1] * filter[filterOffset + 2] : 0.0f;

		sum += (col > 0) ? inputs[inputOffset - 1] * filter[filterOffset + 3] : 0.0f;
		sum += inputs[inputOffset] * filter[filterOffset + 4];
		sum += (col < nbyn - 1) ? inputs[inputOffset + 1] * filter[filterOffset + 5] : 0.0f;

		sum += (col > 0 && row < nbyn - 1) ? inputs[inputOffset + nbyn - 1] * filter[filterOffset + 6] : 0.0f;
		sum += (row < nbyn - 1) ? inputs[inputOffset + nbyn] * filter[filterOffset + 7] : 0.0f;
		sum += (col < nbyn - 1 && row < nbyn - 1) ? inputs[inputOffset + nbyn + 1] * filter[filterOffset + 8] : 0.0f;
	}

	outputs[(outNeuron * offset) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
}