__kernel void conv_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    const int inDim,
    const int nbyn
) {
    // 전역 ID 가져오기
    const int row = get_global_id(0) % nbyn;
    const int col = get_global_id(0) / nbyn;
    const int outNeuron = get_global_id(1);

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;

    float sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = inNeuron * nbyn * nbyn;  // 입력 채널에 대한 offset
        filterOffset = (outNeuron * inDim + inNeuron) * 9; // 필터에 대한 offset
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;  // 필터의 좌우 이동
                y = row + fRow - 1;  // 필터의 상하 이동
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    sum += inputs[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
    }

    // ReLU 적용 및 결과 저장
    outputs[(outNeuron * nbyn * nbyn) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
}


__kernel void pooling_kernel(
	__global float* inputs,
	__global float* outputs,
	const int nbyn
) {
	const int row = get_global_id(0);
	const int col = get_global_id(1);
	const int n = get_global_id(2);

	const int inputNbyn = nbyn * 2;
	const int inputRow = row * 2;
	const int inputCol = col * 2;

	float maxVal = inputs[(n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol];
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol + 1], maxVal);
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol], maxVal);
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol + 1], maxVal);

	outputs[(n * nbyn * nbyn) + (row * nbyn) + col] = maxVal;
}


__kernel void fc_kernel(
	__global float* inputs,
	__global float* outputs,
	__global float* weights,
	__global float* biases,
	const int inDim
) {
	const int outNeuron = get_global_id(0);

	float sum = 0.0f;
	for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		sum += inputs[inNeuron] * weights[outNeuron * inDim + inNeuron];
	}

	outputs[outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}