    for (int i = 0; i < num_of_image; i += batch_size) {
        int current_batch_size = (i + batch_size > num_of_image) ? (num_of_image - i) : batch_size;

        if (i == 0) {

            err = clEnqueueWriteBuffer(read_queue, imageBuf, CL_FALSE, 0,
                sizeof(float) * 3 * 32 * 32 * current_batch_size,
                images + i * 32 * 32 * 3, 0, NULL, &write_event);
            CHECK_ERROR(err);


            convolution_layer(cnn_queue_list[0], conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0],
                INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], current_batch_size,
                &write_event, NULL);

            convolution_layer(cnn_queue_list[0], conv_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1],
                INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[0], pooling_kernel, &layerBuf[1], &layerBuf[2],
                INPUT_DIM[2], NBYN[2], current_batch_size,
                NULL, &pool_event[0]);



            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3],
                INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], current_batch_size,
                &pool_event[0], NULL);

            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4],
                INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[1], pooling_kernel, &layerBuf[4], &layerBuf[5],
                INPUT_DIM[5], NBYN[5], current_batch_size,
                NULL, &pool_event[1]);



            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6],
                INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], current_batch_size,
                &pool_event[1], NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7],
                INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8],
                INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[2], pooling_kernel, &layerBuf[8], &layerBuf[9],
                INPUT_DIM[9], NBYN[9], current_batch_size,
                NULL, &pool_event[2]);



            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10],
                INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], current_batch_size,
                &pool_event[2], NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11],
                INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12],
                INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[3], pooling_kernel, &layerBuf[12], &layerBuf[13],
                INPUT_DIM[13], NBYN[13], current_batch_size,
                NULL, &pool_event[3]);



            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14],
                INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], current_batch_size,
                &pool_event[3], NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15],
                INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16],
                INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[4], pooling_kernel, &layerBuf[16], &layerBuf[17],
                INPUT_DIM[17], NBYN[17], current_batch_size,
                NULL, &pool_event[4]);


            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18],
                INPUT_DIM[18], OUTPUT_DIM[18], current_batch_size,
                &pool_event[4], NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19],
                INPUT_DIM[19], OUTPUT_DIM[19], current_batch_size,
                NULL, NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20],
                INPUT_DIM[20], OUTPUT_DIM[20], current_batch_size,
                NULL, &conv_event);

            save_layer(save_queue, save_kernel, &layerBuf[20], &labels[i], &confidences[i], current_batch_size, &conv_event, &save_event);

        }
        else {

            err = clEnqueueWriteBuffer(read_queue, imageBuf, CL_FALSE, 0,
                sizeof(float) * 3 * 32 * 32 * current_batch_size,
                images + i * 32 * 32 * 3, 1, &pool_event[0], &write_event);
            CHECK_ERROR(err);
            clReleaseEvent(pool_event[0]);


            clWaitForEvents(1, &write_event); 
            convolution_layer(cnn_queue_list[0], conv_kernel, &imageBuf, &layerBuf[0], &wBuf[0], &bBuf[0],
                INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], current_batch_size,
                &pool_event[1], NULL);
            clReleaseEvent(pool_event[1]);
            clReleaseEvent(write_event);

            convolution_layer(cnn_queue_list[0], conv_kernel, &layerBuf[0], &layerBuf[1], &wBuf[1], &bBuf[1],
                INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[0], pooling_kernel, &layerBuf[1], &layerBuf[2],
                INPUT_DIM[2], NBYN[2], current_batch_size,
                NULL, &pool_event[0]);



            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[2], &layerBuf[3], &wBuf[3], &bBuf[3],
                INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], current_batch_size,
                &pool_event[2], NULL);
            clReleaseEvent(pool_event[2]);

            convolution_layer(cnn_queue_list[1], conv_kernel, &layerBuf[3], &layerBuf[4], &wBuf[4], &bBuf[4],
                INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[1], pooling_kernel, &layerBuf[4], &layerBuf[5],
                INPUT_DIM[5], NBYN[5], current_batch_size,
                NULL, &pool_event[1]);



            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[5], &layerBuf[6], &wBuf[6], &bBuf[6],
                INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], current_batch_size,
                &pool_event[3], NULL);
            clReleaseEvent(pool_event[3]);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[6], &layerBuf[7], &wBuf[7], &bBuf[7],
                INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[2], conv_kernel, &layerBuf[7], &layerBuf[8], &wBuf[8], &bBuf[8],
                INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[2], pooling_kernel, &layerBuf[8], &layerBuf[9],
                INPUT_DIM[9], NBYN[9], current_batch_size,
                NULL, &pool_event[2]);


            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[9], &layerBuf[10], &wBuf[10], &bBuf[10],
                INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], current_batch_size,
                &pool_event[4], NULL);
            clReleaseEvent(pool_event[4]);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[10], &layerBuf[11], &wBuf[11], &bBuf[11],
                INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[3], conv_kernel, &layerBuf[11], &layerBuf[12], &wBuf[12], &bBuf[12],
                INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[3], pooling_kernel, &layerBuf[12], &layerBuf[13],
                INPUT_DIM[13], NBYN[13], current_batch_size,
                NULL, &pool_event[3]);


            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[13], &layerBuf[14], &wBuf[14], &bBuf[14],
                INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], current_batch_size,
                &conv_event, NULL);
            clReleaseEvent(conv_event);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[14], &layerBuf[15], &wBuf[15], &bBuf[15],
                INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], current_batch_size,
                NULL, NULL);

            convolution_layer(cnn_queue_list[4], conv_kernel, &layerBuf[15], &layerBuf[16], &wBuf[16], &bBuf[16],
                INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], current_batch_size,
                NULL, NULL);

            max_pooling_layer(cnn_queue_list[4], pooling_kernel, &layerBuf[16], &layerBuf[17],
                INPUT_DIM[17], NBYN[17], current_batch_size,
                NULL, &pool_event[4]);


            clWaitForEvents(1, &pool_event[4]);
            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[17], &layerBuf[18], &wBuf[18], &bBuf[18],
                INPUT_DIM[18], OUTPUT_DIM[18], current_batch_size,
                &save_event, NULL);
            clReleaseEvent(save_event);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[18], &layerBuf[19], &wBuf[19], &bBuf[19],
                INPUT_DIM[19], OUTPUT_DIM[19], current_batch_size,
                NULL, NULL);

            fully_connected_layer(cnn_queue_list[5], fc_kernel, &layerBuf[19], &layerBuf[20], &wBuf[20], &bBuf[20],
                INPUT_DIM[20], OUTPUT_DIM[20], current_batch_size,
                NULL, &conv_event);

            save_layer(save_queue, save_kernel, &layerBuf[20], &labels[i], &confidences[i], current_batch_size, &conv_event, &save_event);
        }

    }
