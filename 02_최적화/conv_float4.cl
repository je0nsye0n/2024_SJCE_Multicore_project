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

    int inputOffset, filterOffset, x, y, inNeuron, fRow;
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
        filterOffset = (outNeuron * inDim + inNeuron) * 9;
        for (fRow = 0; fRow < 3; ++fRow) {
            float4 filterVal, inputVal;
            filterVal.s0 = filter[filterOffset + 3 * fRow + 0];
            filterVal.s1 = filter[filterOffset + 3 * fRow + 1];
            filterVal.s2 = filter[filterOffset + 3 * fRow + 2];
            filterVal.s3 = 0.0f; // Padding to align with float4
            
            x = col - 1;
            y = row + fRow - 1;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                inputVal.s0 = inputs[inputOffset + nbyn * y + x];
            } else {
                inputVal.s0 = 0.0f;
            }
            
            x = col;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                inputVal.s1 = inputs[inputOffset + nbyn * y + x];
            } else {
                inputVal.s1 = 0.0f;
            }

            x = col + 1;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                inputVal.s2 = inputs[inputOffset + nbyn * y + x];
            } else {
                inputVal.s2 = 0.0f;
            }

            inputVal.s3 = 0.0f; // Padding to align with float4

            sum += filterVal * inputVal;
        }
    }

    float final_sum = sum.s0 + sum.s1 + sum.s2 + sum.s3;
    outputs[batch_idx * outDim * nbyn * nbyn + (outNeuron * nbyn * nbyn) + (row * nbyn) + col] = fmax(final_sum + biases[outNeuron], 0.0f);
}
