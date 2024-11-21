#define TILE_SIZE 8

__kernel void conv_kernel_tiling(
	__global float* inputs,
	__global float* outputs,
	__global float* filter,
	__global float* biases,
	const int inDim,
	const int outDim,
	const int nbyn) {

	int global_row = get_group_id(0) * TILE_SIZE + get_local_id(0);
	int global_col = get_group_id(1) * TILE_SIZE + get_local_id(1);

	int local_row = get_local_id(0);
	int local_col = get_local_id(1);

	__local float local_inputs[TILE_SIZE + 2][TILE_SIZE + 2]; 
	__local float local_filters[3][3]; 
	float sum;

	for (int outChannel = 0; outChannel < outDim; ++outChannel) {
		sum = 0.0f;

		for (int inChannel = 0; inChannel < inDim; ++inChannel) {
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					int load_row = global_row + i;
					int load_col = global_col + j;

					if (load_row >= 0 && load_row < nbyn && load_col >= 0 && load_col < nbyn) {
						int inputIndex = (inChannel * nbyn * nbyn) + (load_row * nbyn) + load_col;
						local_inputs[local_row + 1 + i][local_col + 1 + j] = inputs[inputIndex];
					}
					else {
						local_inputs[local_row + 1 + i][local_col + 1 + j] = 0.0f;
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (local_row < 3 && local_col < 3) {
				int filterIndex = (outChannel * inDim * 9) + (inChannel * 9) + (local_row * 3) + local_col;
				local_filters[local_row][local_col] = filter[filterIndex];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int fRow = 0; fRow < 3; ++fRow) {
				for (int fCol = 0; fCol < 3; ++fCol) {
					sum += local_inputs[local_row + fRow][local_col + fCol] * local_filters[fRow][fCol];
				}
			}
			//barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (global_row < nbyn && global_col < nbyn) {
			int outputIndex = (outChannel * nbyn * nbyn) + (global_row * nbyn) + global_col;
			float local_bias = biases[outChannel];
			outputs[outputIndex] = fmax(0.0f, sum + local_bias);
		}
		//barrier(CLK_LOCAL_MEM_FENCE);
	}
}
