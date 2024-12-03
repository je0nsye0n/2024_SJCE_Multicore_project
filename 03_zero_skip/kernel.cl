__kernel void conv_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    const int inDim,
    const int nbyn,
    const int batch_size,
    const int outDim,
    __global char *bit_check1,
    __global char *bit_check2
) {
    const int batch_idx = get_global_id(1);
    const int flat_idx = get_global_id(0);

    const int outNeuron = flat_idx / (nbyn * nbyn);
    const int row = (flat_idx % (nbyn * nbyn)) / nbyn;
    const int col = flat_idx % nbyn;

    int inputOffset;
    int filterOffset, x, y, inNeuron, fRow, fCol;
    float sum = 0.0f, tmp;
    float tmp2;

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;

                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    if (bit_check1 != NULL && bit_check1[inputOffset + nbyn * y + x] == '1') {
                        sum += inputs[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                    }
                    else {
                        sum += inputs[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                    }
                }

            }
        }
    }
    
    tmp2 = fmax(sum + biases[outNeuron], 0.0f);
    outputs[batch_idx * outDim * nbyn * nbyn + (outNeuron * nbyn * nbyn) + (row * nbyn) + col] = tmp2;
    if (bit_check2 != NULL && tmp2 != 0) {
        bit_check2[batch_idx * outDim * nbyn * nbyn + (outNeuron * nbyn * nbyn) + (row * nbyn) + col] = '1'; // 비트 체크 !!! 0이 아닌 경우에만 1로 바꿔주기
        //printf("%c\n", bit_check2[batch_idx * outDim * nbyn * nbyn + (outNeuron * nbyn * nbyn) + (row * nbyn) + col]);
    }
}

__kernel void winograd_conv_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    const int inDim,
    const int nbyn,
    const int batch_size,
    const int outDim
) {
    const int batch_idx = get_global_id(1);
    const int flat_idx = get_global_id(0);

    const int outNeuron = flat_idx / (nbyn * nbyn);
    const int row = (flat_idx % (nbyn * nbyn)) / nbyn;
    const int col = flat_idx % nbyn;

    int inputOffset;
    int filterOffset, x, y, inNeuron, fRow, fCol;
    float sum = 0.0f, tmp, tmp2;

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;
                tmp = inputs[inputOffset + nbyn * y + x];
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn && tmp != 0) {
                    sum += tmp * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
    }
    outputs[batch_idx * outDim * nbyn * nbyn + (outNeuron * nbyn * nbyn) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
}



__kernel void pooling_kernel(
    __global float* inputs,
    __global float* outputs,
    const int nbyn,
    const int batch_size,
    const int dim
) {
    const int batch_idx = get_global_id(1);
    const int flat_idx = get_global_id(0);

    const int n = flat_idx / (nbyn * nbyn);
    const int row = (flat_idx % (nbyn * nbyn)) / nbyn;
    const int col = flat_idx % nbyn;

    const int inputNbyn = nbyn * 2;
    const int inputRow = row * 2;
    const int inputCol = col * 2;

    float maxVal = inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol];
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol + 1], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol + 1], maxVal);

    outputs[batch_idx * dim * nbyn * nbyn + (n * nbyn * nbyn) + (row * nbyn) + col] = maxVal;
}

__kernel void fused_conv_pool_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    __local float* local_data,
    const int inDim,
    const int nbyn,
    const int batch_size,
    const int outDim
) {
    const int batch_idx = get_group_id(1);
    const int flat_idx = get_local_id(0);
    const int local_size = get_local_size(0);

    const int outNeuron = flat_idx / (nbyn * nbyn);
    const int row = (flat_idx % (nbyn * nbyn)) / nbyn;
    const int col = flat_idx % nbyn;

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float sum = 0.0f;

    // Load input data into local memory
    for (int i = flat_idx; i < nbyn * nbyn; i += local_size) {
        if (i < inDim * nbyn * nbyn) {
            local_data[i] = inputs[batch_idx * inDim * nbyn * nbyn + i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convolution Operation
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = inNeuron * nbyn * nbyn;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    sum += local_data[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
    }
    sum = fmax(sum + biases[outNeuron], 0.0f);

    // Store convolution result in local memory
    local_data[flat_idx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pooling Operation (2x2 Max Pooling)
    if (row % 2 == 0 && col % 2 == 0) {
        const int inputRow = row;
        const int inputCol = col;

        // 이전 컨볼루션 결과를 참조하여 풀링 수행
        float maxVal = local_data[flat_idx];

        if (inputCol + 1 < nbyn) {
            maxVal = fmax(local_data[flat_idx + 1], maxVal);
        }
        if (inputRow + 1 < nbyn) {
            maxVal = fmax(local_data[flat_idx + nbyn], maxVal);
        }
        if ((inputRow + 1 < nbyn) && (inputCol + 1 < nbyn)) {
            maxVal = fmax(local_data[flat_idx + nbyn + 1], maxVal);
        }

        const int pooledRow = row / 2;
        const int pooledCol = col / 2;
        outputs[batch_idx * outDim * (nbyn / 2) * (nbyn / 2) + (outNeuron * (nbyn / 2) * (nbyn / 2)) + (pooledRow * (nbyn / 2)) + pooledCol] = maxVal;
    }
}

__kernel void fc_kernel(
    __global float* inputs,
    __global float* outputs,
    __global float* weights,
    __global float* biases,
    const int inDim,
    const int batch_size,
    const int outputDim
) {
    const int batch_idx = get_global_id(1);
    const int outNeuron = get_global_id(0);

    float sum = 0.0f;
    float tmp;

    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        sum += inputs[batch_idx * inDim + inNeuron] * weights[outNeuron * inDim + inNeuron];
    }

    outputs[batch_idx * outputDim + outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}