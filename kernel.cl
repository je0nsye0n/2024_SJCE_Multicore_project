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
    const int batch_idx = get_global_id(2);
    const int row = get_global_id(0) / nbyn;
    const int col = get_global_id(0) % nbyn;
    const int outNeuron = get_global_id(1);

    int inputOffset, filterOffset, x, y, inNeuron, fRow, fCol;
    float sum = 0.0f;

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
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
