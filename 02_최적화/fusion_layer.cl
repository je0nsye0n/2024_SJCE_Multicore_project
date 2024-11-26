__kernel void fusion_kernel(
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
    const int outNeuron = flat_idx / ((nbyn / 2) * (nbyn / 2));
    const int out_row = (flat_idx % ((nbyn / 2) * (nbyn / 2))) / (nbyn / 2);
    const int out_col = flat_idx % (nbyn / 2);

    int inputOffset, filterOffset, x, y;
    float max_val = 0.0f, tmp;

    // Flattened loop to iterate over the 2x2 pooling window and convolution window
    for (int window_idx = 0; window_idx < 4; ++window_idx) {
        int pRow = window_idx / 2;
        int pCol = window_idx % 2;
        float sum = 0.0f;

        // Convolution computation for each pooling window
        for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
            inputOffset = batch_idx * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
            filterOffset = (outNeuron * inDim + inNeuron) * 9;

            for (int f_idx = 0; f_idx < 9; ++f_idx) {
                int fRow = f_idx / 3;
                int fCol = f_idx % 3;

                x = (out_col * 2 + pCol) + fCol - 1;
                y = (out_row * 2 + pRow) + fRow - 1;

                tmp = inputs[inputOffset + nbyn * y + x];
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    sum += inputs[inputOffset + nbyn * y + x] * filter[filterOffset + 3 * fRow + fCol];
                }
            }
        }
        max_val = fmax(max_val, fmax(sum + biases[outNeuron], 0.0f));
    }
    int output_idx = batch_idx * outDim * (nbyn / 2) * (nbyn / 2) + outNeuron * (nbyn / 2) * (nbyn / 2) + out_row * (nbyn / 2) + out_col;
    outputs[output_idx] = max_val;
}
