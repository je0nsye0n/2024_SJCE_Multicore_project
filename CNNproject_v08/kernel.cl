__kernel void conv_kernel(
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

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float8 sumVec = 0.0f;
    float sum = 0.0f;

    float8 inputVec, filterVec;
    float rest;

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;

        for (fRow = 0; fRow < 3; ++fRow) {
            for(fCol = 0; fCol < 3; ++fCol){
                x = col + fCol - 1;
                y = row + fRow - 1;
                if(fRow == 2 && fCol == 2){
                    if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                        rest = inputs[inputOffset + y * nbyn + x];
                    } else {
                        rest = 0.0f;
                    }
                    break;
                }
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    inputVec[fRow * 3 + fCol] = inputs[inputOffset + y * nbyn + x];
                } else {
                    inputVec[fRow * 3 + fCol] = 0.0f;
                }
            }
        }
        if(any(inputVec != 0)) {
            filterVec = vload8(0, filter + filterOffset);
            sumVec = fma(inputVec, filterVec, sumVec);
        }
        if(rest != 0){
            sum += rest * filter[filterOffset + 8];
        }
    }

    for(int i = 0; i < 8; ++i){
        sum += sumVec[i];
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
    const int global_flat_idx = get_global_id(0);
    
    const int n = global_flat_idx / (nbyn * nbyn);
    const int spatial_idx = global_flat_idx % (nbyn * nbyn);
    const int row = spatial_idx / nbyn;
    const int col = spatial_idx % nbyn;

    const int inputNbyn = nbyn * 2;
    const int inputRow = row * 2;
    const int inputCol = col * 2;

    float maxVal = inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol];
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol + 1], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol + 1], maxVal);

    outputs[batch_idx * dim * nbyn * nbyn + (n * nbyn * nbyn) + spatial_idx] = maxVal;
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

    float16 sumVec = 0.0f;
    
    float16 inputVec, weightVec;
    
    const int inputOffset = batch_idx * inDim;
    const int weightOffset = outNeuron * inDim;

    for (int inNeuron = 0; inNeuron < inDim / 16; ++inNeuron) {
        inputVec = vload16(inNeuron, inputs + inputOffset);
        if(any(inputVec != 0)) {
            weightVec = vload16(inNeuron, weights + weightOffset);
            sumVec = fma(inputVec, weightVec, sumVec);
        }
    }
    
    float sum = 0.0f;
    for(int i = 0; i < 16; ++i){
        sum += sumVec[i];
    }

    outputs[batch_idx * outputDim + outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv_tile_kernel(
    __global float* inputs, 
    __global float* outputs,
    __global float* filter,
    __global float* biases,
    const int inDim,
    const int nbyn,
    const int batch_size,
    const int outDim
) {
    const int spatial_idx = get_global_id(0);
    const int outNeuron = get_global_id(1);
    const int batch_idx = get_global_id(2);

    const int row = spatial_idx / nbyn;
    const int col = spatial_idx % nbyn;
    
    const int l_spatial_idx = get_local_id(0);
    const int l_c = get_local_id(1);
    const int wgSize = get_local_size(1);
    
    __local float8 l_inputVec[256];
    __local float l_rest[256];
    float8 sumVec = 0.0f;
    float sum = 0.0f;
    
    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol, i, j, k;

    int localInputOffset = l_spatial_idx * wgSize;
    filterOffset = outNeuron * inDim * 9;
    float8 inputVec, filterVec;
    float tmp;

    for (i = 0; i < inDim; i += wgSize) {
        inNeuron = i + l_c;

        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;

        for (fRow = 0; fRow < 3; ++fRow) {
            for(fCol = 0; fCol < 3; ++fCol){
                x = col + fCol - 1;
                y = row + fRow - 1;
                if(fRow == 2 && fCol == 2){
                    if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                        l_rest[localInputOffset + l_c] = inputs[inputOffset + y * nbyn + x];
                    } else {
                        l_rest[localInputOffset + l_c] = 0.0f;
                    }
                    break;
                }
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    inputVec[fRow * 3 + fCol] = inputs[inputOffset + y * nbyn + x];
                } else {
                    inputVec[fRow * 3 + fCol] = 0.0f;
                }
            }
        }
        l_inputVec[localInputOffset + l_c] = inputVec;

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = 0; j < wgSize; ++j) {
            int filterIdx = filterOffset + (i + j) * 9;
            inputVec = l_inputVec[localInputOffset + j];
            if(any(inputVec != 0)) {
                filterVec = vload8(0, filter + filterIdx);
                sumVec = fma(inputVec, filterVec, sumVec);
            }
            tmp = l_rest[localInputOffset + j];
            if(tmp != 0){
                sum += tmp * filter[filterIdx + 8];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for(int i = 0; i < 8; ++i){
        sum += sumVec[i];
    }

    outputs[batch_idx * outDim * nbyn * nbyn + outNeuron * nbyn * nbyn + spatial_idx] = fmax(sum + biases[outNeuron], 0.0f);
}


