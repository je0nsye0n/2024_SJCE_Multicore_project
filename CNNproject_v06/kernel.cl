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

    int inputOffset, filterOffset, x, y, inNeuron, fRow;
    float sum = 0.0f;

    float3 inputVec, filterVec;

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
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
                filterVec = vload3(fRow, filter + filterOffset);
                sum += dot(inputVec, filterVec);
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

    float sum = 0.0f;
    
    float4 inputVec, weightVec;
    
    const int inputOffset = batch_idx * inDim;
    const int weightOffset = outNeuron * inDim;

    for (int inNeuron = 0; inNeuron < inDim / 4; ++inNeuron) {
        inputVec = vload4(inNeuron, inputs + inputOffset);
        if(!all(inputVec == 0)) {
            weightVec = vload4(inNeuron, weights + weightOffset);
            sum += dot(inputVec, weightVec);
        }
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
    
    __local float3 l_inputVec[3 * 256];
    float sum = 0.0f;
    
    int inputOffset, filterOffset, x, y, inNeuron, fRow, i, j, k;

    filterOffset = outNeuron * inDim * 9;
    float3 inputVec, filterVec;

    for (i = 0; i < inDim; i += wgSize) {
        inNeuron = i + l_c;

        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;

        for (fRow = 0; fRow < 3; ++fRow) {
            x = col - 1;
            y = row + fRow - 1;

            if (y >= 0 && y < nbyn) {
                l_inputVec[l_spatial_idx * 3 * wgSize + l_c * 3 + fRow] = (float3)(
                    (x >= 0 && x < nbyn) ? inputs[inputOffset + y * nbyn + x] : 0.0f,
                    (x + 1 >= 0 && x + 1 < nbyn) ? inputs[inputOffset + y * nbyn + x + 1] : 0.0f,
                    (x + 2 >= 0 && x + 2 < nbyn) ? inputs[inputOffset + y * nbyn + x + 2] : 0.0f
                );
            } else {
                l_inputVec[l_spatial_idx * 3 * wgSize + l_c * 3 + fRow] = (float3)(0.0f, 0.0f, 0.0f); 
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = 0; j < wgSize; ++j) {
            int filterIdx = filterOffset + (i + j) * 9;

            for(k = 0; k < 3; ++k){
                inputVec = l_inputVec[l_spatial_idx * 3 * wgSize + j * 3 + k];
                if(!all(inputVec == 0)) {
                    filterVec = vload3(k, filter + filterIdx);
                    sum += dot(inputVec, filterVec);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    outputs[batch_idx * outDim * nbyn * nbyn + outNeuron * nbyn * nbyn + spatial_idx] = fmax(sum + biases[outNeuron], 0.0f);
}
