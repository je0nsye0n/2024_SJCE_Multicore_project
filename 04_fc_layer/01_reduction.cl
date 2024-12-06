
#define FC_BLOCK_SIZE 64
__kernel void fc_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* weights,
	__global float* biases,
	const int inDim,
	const int outDim
) {
	const int batchIdx = get_group_id(1), outNeuron = get_group_id(0);
	const int tileIdx = get_local_id(0);
	weights += outNeuron * inDim, inputs += batchIdx * inDim;
	float sum = 0.0f;

	for (int i = tileIdx; i < inDim; i += FC_BLOCK_SIZE) {
		sum += inputs[i] * weights[i];
	}

	__local float l_sum[FC_BLOCK_SIZE];
	l_sum[tileIdx] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = FC_BLOCK_SIZE / 2; i > 0; i /= 2) {
		if (tileIdx < i) {
			l_sum[tileIdx] += l_sum[tileIdx + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tileIdx == 0) outputs[batchIdx * outDim + outNeuron] = max(l_sum[0] + biases[outNeuron], 0.0f);
}