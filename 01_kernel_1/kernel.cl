__kernel void fc_layer_kernel(__global float* input, __global float* output, __global float* weights, __global float* biases, int inDim, int outDim) {

	int id = get_global_id(0);
	float sum = 0.0f;

	if (id < outDim) {
		for (int i = 0; i < inDim; i++) {
			sum += input[i] * weights[id * inDim + i]; // 출력 뉴런 id가 참조해야할 inDim개의 가중치 중 i번째 가중치임을 의미
		}

		sum += biases[id];

		output[id] = (sum > 0) ? sum : 0.0;
	}
}

__kernel void max_pooling_kernel(__global const float* input, __global float* output, int nbyn) {
    int dim = get_global_id(0); // 현재 채널 인덱스
    int row = get_global_id(1); // 2씩 증가하여 각 2x2 영역을 처리
    int col = get_global_id(2);

    if (row < nbyn && col < nbyn) {
        // 각 2x2 영역의 최대값을 계산
        float max_val = 0.0f;

        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                int input_index = dim * nbyn * nbyn + (row + y) * nbyn + (col + x);
                float temp = input[input_index];
                if (temp > max_val)  max_val = temp;
                
            }
        }

        // 출력 배열에 최대값 저장
        int output_index = dim * (nbyn / 2) * (nbyn / 2) + (row / 2) * (nbyn / 2) + (col / 2);
        output[output_index] = max_val;
    }
}
