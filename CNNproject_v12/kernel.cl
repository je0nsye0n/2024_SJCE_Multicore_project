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

	int inputOffset, filterOffset, x, y, fRow;
	float3 sum = 0.0f;

	float3 inputVec, filterVec;
	
	inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
	filterOffset = totalDim * 9;	

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

	outputs[((batchIdx * outDim + outNeuron) * inDim + inNeuron) * featureMapSize + spatialIdx] = sum.x + sum.y + sum.z;
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

	int inputOffset, x, y, inNeuron, fRow;
	float sum = 0.0f;

	inputOffset = (batchIdx * outDim + outNeuron) * inDim;

	for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		sum += inputs[(inputOffset + inNeuron) * featureMapSize + spatialIdx];
	}

	outputs[(batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx] = fmax(sum + biases[outNeuron], 0.0f);
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

	float maxVal;

	maxVal = inputs[(inputOffset + inputRow) * inputNbyn + inputCol];
	maxVal = fmax(inputs[(inputOffset + inputRow) * inputNbyn + inputCol + 1], maxVal);
	maxVal = fmax(inputs[(inputOffset + (inputRow + 1)) * inputNbyn + inputCol], maxVal);
	maxVal = fmax(inputs[(inputOffset + (inputRow + 1)) * inputNbyn + inputCol + 1], maxVal);

	outputs[(batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx] = maxVal;
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

	float sum, tmp;

	sum = 0.0f;
	for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		tmp = inputs[inputOffset + inNeuron];
		if (tmp != 0) {
			sum += tmp * weights[weightOffset + inNeuron];
		}
	}

	outputs[batchIdx * outDim + outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}