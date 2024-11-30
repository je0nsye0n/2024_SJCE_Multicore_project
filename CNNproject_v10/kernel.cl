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

    const int outNeuron = flatIdx / (nbyn * nbyn);
    const int spatialIdx = flatIdx % (nbyn * nbyn);
    const int row = spatialIdx / nbyn;
    const int col = spatialIdx % nbyn;

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float sum, tmp;

    sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = (batchIdx * inDim + inNeuron) * nbyn * nbyn;
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

    outputs[(batchIdx * outDim + outNeuron) * nbyn * nbyn + spatialIdx] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv1_kernel(
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

    const int l_spatialIdx = get_local_id(0);
    const int l_size = get_local_id(0);
    const int l_row = l_spatialIdx / nbyn;
    const int l_col = l_spatialIdx % nbyn;
    const int l_maxRow = l_size / nbyn; 

    __local float l_inputs[64];
    __local float l_filter[9];

    int inputOffset, filterOffset, x, y, l_x, l_y, inNeuron, fRow, fCol;
    float sum;

    sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;

        l_inputs[l_spatialIdx] = inputs[inputOffset + spatialIdx];
        if(l_spatialIdx < 9) {
            l_filter[l_spatialIdx] = filter[filterOffset + l_spatialIdx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    l_x = l_col + fCol - 1;
                    l_y = l_row + fRow - 1;
                    if (l_x >= 0 && l_x < nbyn && l_y >= 0 && l_y < l_maxRow) {
                        sum += l_inputs[l_y * nbyn + l_x] * l_filter[3 * fRow + fCol];
                    }
                    else {
                        sum += inputs[inputOffset + y * nbyn + x] * l_filter[3 * fRow + fCol];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    outputs[(batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv2_kernel(
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

    __local float l_inputs[64];
    __local float l_filter[9];

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float sum;

    sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;

        l_inputs[spatialIdx] = inputs[inputOffset + spatialIdx];
        if(spatialIdx < 9) {
            l_filter[spatialIdx] = filter[filterOffset + spatialIdx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (fRow = 0; fRow < 3; ++fRow) {
            for (fCol = 0; fCol < 3; ++fCol) {
                x = col + fCol - 1;
                y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    sum += l_inputs[y * nbyn + x] * l_filter[fRow * 3 + fCol];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    outputs[(batchIdx * outDim + outNeuron) * featureMapSize + spatialIdx] = fmax(sum + biases[outNeuron], 0.0f);
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
    float sum, tmp;

    sum = 0.0f;
    for (inNeuron = 0; inNeuron < inDim; inNeuron += l_channel) {
        inputOffset = (batchIdx * inDim + inNeuron) * featureMapSize;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;

        l_inputs[l_flatIdx] = inputs[inputOffset + l_flatIdx];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ch = 0; ch < l_channel; ++ch) {
            for (fRow = 0; fRow < 3; ++fRow) {
                for (fCol = 0; fCol < 3; ++fCol) {
                    x = col + fCol - 1;
                    y = row + fRow - 1;
                    if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                        tmp = l_inputs[ch * featureMapSize + y * nbyn + x];
                        if(tmp != 0) {
                            sum += tmp * filter[filterOffset + ch * 9 + fRow * 3 + fCol];
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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
        if(tmp != 0) {
            sum += tmp * weights[weightOffset + inNeuron];
        }
    }

    outputs[batchIdx * outDim + outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}