__kernel void im2col(
    __global float* inputs,
    __global float* outputs,
    const int inDim,
    const int nbyn,
    const int batch_size
) {
    const int batch_idx = get_global_id(1);
    const int flat_idx = get_global_id(0);

    const int inNeuron = flat_idx / (nbyn * nbyn);
    const int row = (flat_idx % (nbyn * nbyn)) / nbyn;
    const int col = flat_idx % nbyn;

    int inputOffset, outputOffset, x, y, fRow, fCol;

    inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
    outputOffset = batch_idx * inDim * nbyn * nbyn * 9 + (row * nbyn + col) * 9 * inDim + inNeuron * 9;
    for (fRow = 0; fRow < 3; ++fRow) {
        for (fCol = 0; fCol < 3; ++fCol) {
            x = col + fCol - 1;
            y = row + fRow - 1;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                outputs[outputOffset + fRow * 3 + fCol] = inputs[inputOffset + nbyn * y + x];
            }
            else{
                outputs[outputOffset + fRow * 3 + fCol] = 0.0f;
            }
        }
    }
}

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

    int inputOffset = batch_idx * inDim * nbyn * nbyn * 9 + (row * nbyn + col) * 9 * inDim;
    int filterOffset = outNeuron * inDim * 9;


    float sum = 0.0f;
    for (int i = 0; i < inDim * 9; ++i) {
        sum += inputs[inputOffset + i] * filter[filterOffset + i];
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
    const int batch_idx = get_global_id(2);
    const int row = get_global_id(0) / nbyn;
    const int col = get_global_id(0) % nbyn;
    const int n = get_global_id(1);

    const int inputNbyn = nbyn * 2;
    const int inputRow = row * 2;
    const int inputCol = col * 2;

    float maxVal = inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol];
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol + 1], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol], maxVal);
    maxVal = fmax(inputs[batch_idx * dim * inputNbyn * inputNbyn + (n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol + 1], maxVal);

    outputs[batch_idx * dim * nbyn * nbyn + (n * nbyn * nbyn) + (row * nbyn) + col] = maxVal;
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
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        sum += inputs[batch_idx * inDim + inNeuron] * weights[outNeuron * inDim + inNeuron];
    }

    outputs[batch_idx * outputDim + outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}