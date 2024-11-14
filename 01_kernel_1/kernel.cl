__kernel void fc_layer_kernel(__global float* input, __global float* output, __global float* weights, __global float* biases, int inDim, int outDim) {

	int id = get_global_id(0);
	float sum = 0.0f;

	if (id < outDim) {
		for (int i = 0; i < inDim; i++) {
			sum += input[i] * weights[id * inDim + i]; // ��� ���� id�� �����ؾ��� inDim���� ����ġ �� i��° ����ġ���� �ǹ�
		}

		sum += biases[id];

		output[id] = (sum > 0) ? sum : 0.0;
	}
}

__kernel void max_pooling_kernel(__global const float* input, __global float* output, int nbyn) {
    int dim = get_global_id(0); // ���� ä�� �ε���
    int row = get_global_id(1); // 2�� �����Ͽ� �� 2x2 ������ ó��
    int col = get_global_id(2);

    if (row < nbyn && col < nbyn) {
        // �� 2x2 ������ �ִ밪�� ���
        float max_val = 0.0f;

        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                int input_index = dim * nbyn * nbyn + (row + y) * nbyn + (col + x);
                float temp = input[input_index];
                if (temp > max_val)  max_val = temp;
                
            }
        }

        // ��� �迭�� �ִ밪 ����
        int output_index = dim * (nbyn / 2) * (nbyn / 2) + (row / 2) * (nbyn / 2) + (col / 2);
        output[output_index] = max_val;
    }
}
