__kernel void conv_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* filter,
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

	int inputOffset, filterOffset, x, y, inNeuron, fRow;
	float3 sum = 0.0f;

	float3 inputVec, filterVec;

	for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		inputOffset = (batchIdx * inDim + inNeuron) * nbyn * nbyn;
		filterOffset = (outNeuron * inDim + inNeuron) * 9;

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
	}

	outputs[(batchIdx * outDim + outNeuron) * nbyn * nbyn + spatialIdx] = fmax(sum.x + sum.y + sum.z + biases[outNeuron], 0.0f);
}

__kernel void conv3_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* filter,
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

	const int l_flatIdx = get_local_id(0);
	const int l_size = get_local_size(0);
	const int l_inNeuron = l_flatIdx / featureMapSize;
	const int l_spatialIdx = l_flatIdx % featureMapSize;
	const int l_channel = l_size / featureMapSize;

	__local float l_inputs[64];

	int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;

	float3 sum = 0.0f;

	float3 inputVec, filterVec;

	sum = 0.0f;
	for (inNeuron = 0; inNeuron < inDim; inNeuron += l_channel) {
		inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
		filterOffset = (outNeuron * inDim + inNeuron) * 9;

		l_inputs[l_flatIdx] = inputs[inputOffset + l_flatIdx];

		barrier(CLK_LOCAL_MEM_FENCE);
		for (int ch = 0; ch < l_channel; ++ch) {
			for (fRow = 0; fRow < 3; ++fRow) {
				x = col - 1;
				y = row + fRow - 1;
				if (y >= 0 && y < nbyn) {
					inputVec = (float3)(
						(x >= 0 && x < nbyn) ? l_inputs[ch * featureMapSize + y * nbyn + x] : 0.0f,
						(x + 1 >= 0 && x + 1 < nbyn) ? l_inputs[ch * featureMapSize + y * nbyn + x + 1] : 0.0f,
						(x + 2 >= 0 && x + 2 < nbyn) ? l_inputs[ch * featureMapSize + y * nbyn + x + 2] : 0.0f
					);
					if (any(inputVec != (float3)(0.0f, 0.0f, 0.0f))) {
						filterVec = vload3(fRow, filter + filterOffset + ch * 9);
						sum = fma(inputVec, filterVec, sum);
					}
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	outputs[(batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx] = fmax(sum.x + sum.y + sum.z + biases[outNeuron], 0.0f);
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
