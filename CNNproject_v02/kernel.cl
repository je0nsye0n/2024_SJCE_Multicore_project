__kernel void conv_kernel ( 
	__global float *inputs,
	__global float *outputs,
	__global float *filter,
	__global float *biases,
	const int inDim,
	const int nbyn
) {
	const int row = get_global_id(0) / nbyn;
	const int col = get_global_id(0) % nbyn;
	const int outNeuron = get_global_id(1);
	
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

__kernel void pooling_kernel ( 
	__global float *inputs,
	__global float *outputs,
	const int nbyn
) {
	const int row = get_global_id(0) / nbyn;
	const int col = get_global_id(0) % nbyn;
	const int n = get_global_id(1);

	const int inputNbyn = nbyn * 2;
	const int inputRow = row * 2;
	const int inputCol = col * 2;

	float maxVal = inputs[(n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol];
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + (inputRow * inputNbyn) + inputCol + 1], maxVal);
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol], maxVal);
	maxVal = fmax(inputs[(n * inputNbyn * inputNbyn) + ((inputRow + 1) * inputNbyn) + inputCol + 1], maxVal);

	outputs[(n * nbyn * nbyn) + (row * nbyn) + col] = maxVal;
}


__kernel void fc_kernel ( 
	__global float *inputs,
	__global float *outputs,
	__global float *weights,
	__global float *biases,
	const int inDim
) {
	const int outNeuron = get_global_id(0);
	
	float sum = 0.0f;
	for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
		sum += inputs[inNeuron] * weights[outNeuron * inDim + inNeuron];
	}
	
	outputs[outNeuron] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv_lm_kernel ( 
    __global float *inputs,
    __global float *outputs,
    __global float *filter,
    __global float *biases,
    const int inDim,
    const int nbyn
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int outNeuron = get_global_id(2);
    
    const int l_row = get_local_id(0);
    const int l_col = get_local_id(1);
    const int l_ch = get_local_id(2); 
    const int wgSize = get_local_size(2); 

    __local float l_input[4][4][32];
    
    const int offset = nbyn * nbyn;
	int i, j, inNeuron, inputOffset, filterOffset, fRow, fCol, x, y;
    
    float sum = 0.0f;
    for (i = 0; i < inDim; i += wgSize) {
        inNeuron = i + l_ch;
        inputOffset = inNeuron * offset;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (fRow = l_row; fRow < 2 + l_row; ++fRow) {
			for (fCol = l_col; fCol < 2 + l_col; ++fCol) {
		        x = col + fCol - 1;
	            y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn)
					l_input[l_row + fRow][l_col + fCol][l_ch] = inputs[inputOffset + nbyn * y + x];
				else
			        l_input[l_row + fRow][l_col + fCol][l_ch] = 0.0f;
	        }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = 0; j < wgSize; ++j) {
			filterOffset = (outNeuron * inDim + i + j) * 9;
			for (fRow = 0; fRow < 3; ++fRow) {
		        for (fCol = 0; fCol < 3; ++fCol) {
	                sum += l_input[l_row + fRow][l_col + fCol][j] * filter[filterOffset + fRow * 3 + fCol];
				}
			}
        }
    }
    
    outputs[(outNeuron * offset) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
}

__kernel void conv_lm1_kernel ( 
    __global float *inputs,
    __global float *outputs,
    __global float *filter,
    __global float *biases,
    const int inDim,
    const int nbyn
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int outNeuron = get_global_id(2);
    
    const int l_ch = get_local_id(2); 
    const int wgSize = get_local_size(2); 
    __local float l_input[3][3][32];
    
    const int offset = nbyn * nbyn;
	int i, j, inNeuron, inputOffset, filterOffset, fRow, fCol, x, y;
    
    float sum = 0.0f;
    for (i = 0; i < inDim; i += wgSize) {
        inNeuron = i + l_ch;
        
        inputOffset = inNeuron * offset;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int fRow = 0; fRow < 3; ++fRow) {
			for (int fCol = 0; fCol < 3; ++fCol) {
		        int x = col + fCol - 1;
	            int y = row + fRow - 1;
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn)
					l_input[fRow][fCol][l_ch] = inputs[inputOffset + nbyn * y + x];
				else
			        l_input[fRow][fCol][l_ch] = 0.0f;
	        }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = 0; j < wgSize; ++j) {
			filterOffset = (outNeuron * inDim + i + j) * 9;
			for (fRow = 0; fRow < 3; ++fRow) {
		        for (fCol = 0; fCol < 3; ++fCol) {
	                sum += l_input[fRow][fCol][j] * filter[filterOffset + fRow * 3 + fCol];
				}
			}
        }
    }
    
    outputs[(outNeuron * offset) + (row * nbyn) + col] = fmax(sum + biases[outNeuron], 0.0f);
}
