
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
	const int vectorSize = 4; // Using float4

	// Adjust pointers for the specific output neuron and batch
	weights += outNeuron * inDim;
	inputs += batchIdx * inDim;

	float sum = 0.0f;
	float4 inputVec, weightVec;

	// Iterate over the input dimensions in chunks of vectorSize (float4)
	int i;
	for (i = tileIdx * vectorSize; i < inDim - (vectorSize - 1); i += FC_BLOCK_SIZE * vectorSize) {
		inputVec = vload4(0, inputs + i);
		weightVec = vload4(0, weights + i);

		// Perform element-wise multiplication and accumulate
		sum += dot(inputVec, weightVec);
	}

	// Handle remaining elements if inDim is not a multiple of vectorSize
	for (; i < inDim; i++) {
		sum += inputs[i] * weights[i];
	}

	// Use local memory to accumulate results from different work-items
	__local float l_sum[FC_BLOCK_SIZE];
	l_sum[tileIdx] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform reduction within the workgroup
	for (int stride = FC_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		if (tileIdx < stride) {
			l_sum[tileIdx] += l_sum[tileIdx + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write the final result for this output neuron
	if (tileIdx == 0) {
		outputs[batchIdx * outDim + outNeuron] = fmax(l_sum[0] + biases[outNeuron], 0.0f);
	}
}
