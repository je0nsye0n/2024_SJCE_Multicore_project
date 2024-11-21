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

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;

    float sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = inNeuron * offset;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    sum += inputs[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
    }

    outputs[(outNeuron * offset) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
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

__kernel void conv_pool_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    __local float* local_conv,
    const int inDim,
    const int nbyn
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int outNeuron = get_global_id(2);

    const int localRow = get_local_id(0) + 1; // 패딩을 위해 인덱스를 1 증가
    const int localCol = get_local_id(1) + 1; // 패딩을 위해 인덱스를 1 증가
    const int offset = nbyn * nbyn;
    const int poolNbyn = nbyn / 2;

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float sum = 0.0f;

    // Local memory 초기화 (로컬 메모리 접근 범위 수정)
    for (fRow = -1; fRow <= 1; ++fRow) {
        for (fCol = -1; fCol <= 1; ++fCol) {
            x = col + fCol;
            y = row + fRow;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                // Local memory 접근 범위가 적절한지 확인
                if ((localRow + fRow) >= 0 && (localRow + fRow) < (8 + 2) &&
                    (localCol + fCol) >= 0 && (localCol + fCol) < (8 + 2)) {
                    local_conv[(localRow + fRow) * (nbyn + 2) + (localCol + fCol)] = inputs[y * nbyn + x];
                }
            }
            else {
                local_conv[(localRow + fRow) * (nbyn + 2) + (localCol + fCol)] = 0.0f;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convolution 연산 수행
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = inNeuron * offset;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        sum = 0.0f;
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                int localIndexRow = localRow + fRow - 1;
                int localIndexCol = localCol + fCol - 1;

                // Local memory 접근 범위 확인
                if (localIndexRow >= 0 && localIndexRow < (nbyn + 2) &&
                    localIndexCol >= 0 && localIndexCol < (nbyn + 2)) {
                    sum += local_conv[localIndexRow * (nbyn + 2) + localIndexCol] * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
    }

    // ReLU 적용
    float convResult = fmax(sum + biases[outNeuron], 0.0f);

    // Pooling 연산 (2x2 max pooling)
    if ((row % 2 == 0) && (col % 2 == 0)) {
        float maxVal = convResult;

        if ((localCol + 1) < (nbyn + 2)) {
            maxVal = fmax(maxVal, local_conv[localRow * (nbyn + 2) + localCol + 1]);
        }
        if ((localRow + 1) < (nbyn + 2)) {
            maxVal = fmax(maxVal, local_conv[(localRow + 1) * (nbyn + 2) + localCol]);
        }
        if ((localRow + 1) < (nbyn + 2) && (localCol + 1) < (nbyn + 2)) {
            maxVal = fmax(maxVal, local_conv[(localRow + 1) * (nbyn + 2) + localCol + 1]);
        }

        outputs[(outNeuron * poolNbyn * poolNbyn) + ((row / 2) * poolNbyn) + (col / 2)] = maxVal;
    }
}
