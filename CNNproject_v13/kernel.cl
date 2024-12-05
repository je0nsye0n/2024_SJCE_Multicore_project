__kernel void conv_mul_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* filter,
	const int inDim,
	const int outDim,
	const int nbyn
) {
	const int flatIdx = get_global_id(0);
	const int batchIdx = get_global_id(1);

	const int featureMapSize = nbyn * nbyn;
	const int totalDim = flatIdx / featureMapSize;
	const int outNeuron = totalDim / inDim;
	const int inNeuron = totalDim % inDim;
	const int spatialIdx = flatIdx % featureMapSize;
	const int row = spatialIdx / nbyn;
	const int col = spatialIdx % nbyn;

	const int inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
	const int filterOffset = totalDim * 9;	
	const int outputOffset = ((batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx) * inDim + inNeuron;

	float3 inputVec, filterVec;
	float3 sum = 0.0f;
	
	int x, y, fRow;
	
	for (fRow = 0; fRow < 3; ++fRow) {
		x = col - 1;
		y = row + fRow - 1;
		if (y >= 0 && y < nbyn) {
			inputVec = (float3)(
				(x >= 0 && x < nbyn) ? inputs[inputOffset + y * nbyn + x] : 0.0f,
				(x + 1 >= 0 && x + 1 < nbyn) ? inputs[inputOffset + y * nbyn + x + 1] : 0.0f,
				(x + 2 >= 0 && x + 2 < nbyn) ? inputs[inputOffset + y * nbyn + x + 2] : 0.0f
			);
			if (any(inputVec != (float3)(0.0f, 0.0f, 0.0f))) {
				filterVec = vload3(fRow, filter + filterOffset);
				sum = fma(inputVec, filterVec, sum);
			}
		}
	}

	outputs[outputOffset] = sum.x + sum.y + sum.z;
}

__kernel void conv_sum_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* biases,
	const int inDim,
	const int outDim,
	const int nbyn
) {
	const int flatIdx = get_global_id(0);
	const int batchIdx = get_global_id(1);

	const int featureMapSize = nbyn * nbyn;
	const int outNeuron = flatIdx / featureMapSize;
	const int spatialIdx = flatIdx % featureMapSize;
	const int row = spatialIdx / nbyn;
	const int col = spatialIdx % nbyn;

	const int inputOffset = ((batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx) * inDim;
	const int outputOffset = (batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx;
	
	float16 sumVec = 0.0f;
	float sum = 0.0f;

	int inNeuron, i;

	for (inNeuron = 0; inNeuron < inDim / 16; ++inNeuron) {
		sumVec += vload16(inNeuron, inputs + inputOffset);
	}
	for (i = 0; i < 16; ++i) {
		sum += sumVec[i];
	}

	outputs[outputOffset] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv_sum1_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* biases,
	const int inDim,
	const int outDim,
	const int nbyn
) {
	const int flatIdx = get_global_id(0);
	const int batchIdx = get_global_id(1);

	const int featureMapSize = nbyn * nbyn;
	const int outNeuron = flatIdx / featureMapSize;
	const int spatialIdx = flatIdx % featureMapSize;
	const int row = spatialIdx / nbyn;
	const int col = spatialIdx % nbyn;

	const int inputOffset = ((batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx) * inDim;
	const int outputOffset = (batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx;

	float3 sumVec = vload3(0, inputs + inputOffset);
	float sum = sumVec.x + sumVec.y + sumVec.z;

	outputs[outputOffset] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void pooling_kernel(
	__global float* inputs,
	__global float* outputs,
	const int outDim,
	const int nbyn
) {
	const int flatIdx = get_global_id(0);
	const int batchIdx = get_global_id(1);

	const int featureMapSize = nbyn * nbyn;
	const int outNeuron = flatIdx / featureMapSize;
	const int spatialIdx = flatIdx % featureMapSize;
	const int row = spatialIdx / nbyn;
	const int col = spatialIdx % nbyn;

	const int inputNbyn = nbyn * 2;
	const int inputRow = row * 2;
	const int inputCol = col * 2;

	const int inputOffset = (batchIdx * outDim + outNeuron) * inputNbyn;
	const int outputOffset = (batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx;

	float maxVal;

	maxVal = inputs[(inputOffset + inputRow) * inputNbyn + inputCol];
	maxVal = fmax(inputs[(inputOffset + inputRow) * inputNbyn + inputCol + 1], maxVal);
	maxVal = fmax(inputs[(inputOffset + (inputRow + 1)) * inputNbyn + inputCol], maxVal);
	maxVal = fmax(inputs[(inputOffset + (inputRow + 1)) * inputNbyn + inputCol + 1], maxVal);

	outputs[outputOffset] = maxVal;
}

__kernel void fc_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* weights,
	__global float* biases,
	const int inDim,
	const int outDim
) {
	const int outNeuron = get_global_id(0);
	const int batchIdx = get_global_id(1);

	const int inputOffset = batchIdx * inDim;
	const int weightOffset = outNeuron * inDim;
	const int outputOffset = batchIdx * outDim + outNeuron;

	float sum, tmp;

	sum = 0.0f;
	for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		tmp = inputs[inputOffset + inNeuron];
		if (tmp != 0) {
			sum += tmp * weights[weightOffset + inNeuron];
		}
	}

	outputs[outputOffset] = fmax(sum + biases[outNeuron], 0.0f);
}