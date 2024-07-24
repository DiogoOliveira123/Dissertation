import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm

from models import TrainModel
from utils import Precision, Recall, F1Score
from create_dataset import *
from sequence_generator import *
from calc_temp_params import *
from gradCam import *
from save_plot_true_pred import save_plot_true_pred

if __name__ == '__main__':

    random.seed(0)
    state = random.getstate()

    # LOCAL PC Paths
    base_path = r'C:\Users\diman\PycharmProjects\dissertation\GaitEventDetection'
    dataset_path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\RGB_labeling_30Hz_balanced_aligned_v6.xlsx'

    # BIRDLAB Paths
    # base_path = '/home/birdlab/Desktop/WALKIT_SW/code/GaitEventDetection/'
    # dataset_path = '/home/birdlab/Desktop/WALKIT_SW/dataset/RGB_labeling_30Hz_balanced_aligned_v6.xlsx'

    # CLUSTER Paths
    # base_path = '/home/id9480/gaitEvents'
    # dataset_path = '/home/id9480/gaitEvents/dataset/RGB_labeling_30Hz_balanced_aligned_v6.xlsx'

    NUM_TRIALS = 24
    NUM_PARTICIPANTS = 15
    TEST_PARTICIPANTS = ['15', '8', '3']
    VAL_PARTICIPANTS = ['2', '11']

    train_network = True
    gradCam = True
    plot_val_true_pred = False
    plot_test_true_pred = False
    get_temp_params = False

    if not train_network:
        data = Dataset()
        balanced_dataset = data.CreateDataset()
        # train, val, test = data.SplitDataset(NUM_PARTICIPANTS, TEST_PARTICIPANTS, VAL_PARTICIPANTS)

    else:
        # train parameters
        epochs = 100
        batch_size = 32
        num_classes = 3
        seq_len = 3
        width = 224
        height = 224
        num_channels = 3

        type_model = "Conv3DLSTM_V7"

        # Define model parameters
        input_shape = (batch_size, seq_len, width, height, num_channels)

        # Models
        results_prefix = type_model + "_GaitEvents_BS" + str(batch_size)
        print("\nNeural network: {}".format(results_prefix))

        print("Epochs: {}".format(epochs))
        print("Batch Size: {}".format(batch_size))

        training = TrainModel(checkpoint_metric_name="val_F1Score", no_epochs=epochs, base_path=base_path,
                              dataset_path=dataset_path, num_classes=num_classes,
                              fixed_width=224, batch_size=batch_size)

        path_train = []
        path_val = []
        path_test = []
        true_labels = []

        # Split TRAIN, VAL, TEST info from CSV FILE
        train_df, val_df, test_df = SequenceDataGenerator.splitTrainValTest(dataset_path)

        # SEQUENCE GENERATOR
        train_generator = SequenceDataGenerator(train_df, batch_size, augment=True)
        val_generator = SequenceDataGenerator(val_df, batch_size, augment=False)
        test_generator = SequenceDataGenerator(test_df, batch_size, augment=False)

        if get_temp_params:
            # Get gait events
            R_HS = []
            R_TO = []
            L_HS = []
            L_TO = []

            for i in range(1, len(true_labels) - 1):
                if true_labels[i - 1] == 0 and true_labels[i] == -1:
                    R_HS.append(i)
                elif true_labels[i - 1] == 0 and true_labels[i] == 1:
                    L_HS.append(i)
                elif true_labels[i - 1] == -1 and true_labels[i] == 0:
                    R_TO.append(i)
                elif true_labels[i - 1] == 1 and true_labels[i] == 0:
                    L_TO.append(i)

            # R: 0->10 | 39->76 | 100->124 | 145->164 | 187->208 | 233->258
            # L: 10->39 | 76->100 | 124->145 | 164->187 | 208->233 | 258->281
            fs = 30  # camera sampling frequency
            temp_params = TemporalParameters(freq=30)

            # Calculate R and L STEP time and GAI (Gait Asymmetry Index) for STEP
            if R_HS[0] < L_HS[0]:  # right foot first
                R_first_step = round(1 / 30 * R_HS[0], 2)
                L_first_step = round((L_HS[0] - R_HS[0]) * 1 / 30, 2)

                R_step_time = temp_params.get_step_time(R_HS[1:], L_HS)
                R_step_time.insert(0, R_first_step)

                L_step_time = temp_params.get_step_time(L_HS[1:], R_HS[1:])
                L_step_time.insert(0, L_first_step)

                R_step_avg_time = sum(R_step_time) / len(R_step_time)
                L_step_avg_time = sum(L_step_time) / len(L_step_time)
                step_index = temp_params.get_asymm_index(R_step_avg_time, L_step_avg_time)

            else:  # left foot first
                L_first_step = round(1 / 30 * L_HS[0], 2)
                R_first_step = round((R_HS[0] - L_HS[0]) * 1 / 30, 2)

                L_step_time = temp_params.get_step_time(L_HS[1:], R_HS)
                L_step_time.insert(0, L_first_step)

                R_step_time = temp_params.get_step_time(L_HS, R_HS[1:])
                R_step_time.insert(0, R_first_step)

                R_step_avg_time = sum(R_step_time) / len(R_step_time)
                L_step_avg_time = sum(L_step_time) / len(L_step_time)
                step_index = temp_params.get_asymm_index(R_step_avg_time, L_step_avg_time)

            num_steps = len(R_step_time) + len(L_step_time)

            # Calculate R and L STRIDE time and GAI (Gait Asymmetry Index) for STRIDE
            R_stride_time = temp_params.get_stride_time(R_HS)
            L_stride_time = temp_params.get_stride_time(L_HS)
            R_stride_avg_time = sum(R_stride_time) / len(R_stride_time)
            L_stride_avg_time = sum(L_stride_time) / len(L_stride_time)
            stride_index = temp_params.get_asymm_index(R_stride_avg_time, L_stride_avg_time)

            # Calculate R and L STANCE time and GAI (Gait Asymmetry Index) for STANCE
            R_stance_time = temp_params.get_stance_time(R_TO, R_HS)
            L_stance_time = temp_params.get_stance_time(L_TO, L_HS)
            R_stance_avg_time = sum(R_stance_time) / len(R_stance_time)
            L_stance_avg_time = sum(L_stance_time) / len(L_stance_time)
            stance_index = temp_params.get_asymm_index(R_stride_avg_time, L_stride_avg_time)

            # Calculate R and L SWING time and GAI (Gait Asymmetry Index) for SWING
            R_swing_time = temp_params.get_swing_time(R_HS[1:], R_TO)
            L_swing_time = temp_params.get_swing_time(L_HS[1:], L_TO)
            R_swing_avg_time = sum(R_swing_time) / len(R_swing_time)
            L_swing_avg_time = sum(L_swing_time) / len(L_swing_time)
            swing_index = temp_params.get_asymm_index(R_swing_avg_time, L_swing_avg_time)

        # Get GradCam images
        if gradCam:
            model = tf.keras.models.load_model('conv3dlstm_V7_model.h5',
                                               custom_objects={'F1Score': F1Score})
            batch_idx = 2
            input_volume = test_generator[batch_idx][0]  # Batch of sequences of frames
            layer_name = 'conv3d_2'  # Layer to apply visualization
            sequence_idx = 0

            # Remove last layer's activation
            model.layers[-1].activation = None
            heatmaps = []

            n_rows = 4
            n_cols = 4

            # Display 4 rows and 2 columns of images
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 16))  # Adjusted figsize for clarity

            for i in range(n_rows):
                for j in range(n_cols):
                    index = sequence_idx + (i * 4 + j) * 2  # Indexing to skip 2 images each iteration
                    if index >= len(input_volume):
                        break

                    # Prepare image array for the model
                    img_array = np.expand_dims(input_volume[index], axis=0)

                    # Print the top predicted class
                    preds = model.predict(img_array)
                    print('Predicted:', preds[0])

                    # Generate class activation heatmap
                    heatmap = obtain_grad_cam2(img_array, model, layer_name)

                    # Resize heatmap to match the input image dimensions
                    resized_heatmap = get_resized_heatmap(heatmap, input_volume.shape)

                    # Plot original image
                    ax[i][j].imshow(np.squeeze(input_volume[index][0]), cmap='bone')
                    img0 = ax[i][j].imshow(np.squeeze(input_volume[index][0]), cmap='bone')
                    # Plot heatmap on top of the image
                    img1 = ax[i][j].imshow(resized_heatmap[:, :],
                                           cmap='jet',
                                           alpha=0.4,
                                           extent=img0.get_extent())
                    ax[i][j].axis('off')

            plt.savefig(r'heatmaps\heatmap_4.png')
            plt.tight_layout()
            plt.show()

        # Build model
        model = training.build_model(type_model=type_model, weights=None, input_shape=input_shape)

        model.summary()

        print('\nCompiling....')
        learning_rate = 0.0001
        decay = 0.005
        momentum = 0.9
        nesterov = True

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            weight_decay=decay,
            momentum=momentum,
            nesterov=nesterov
        )

        print("Learning rate: {}".format(learning_rate))
        print("Decay: {}".format(decay))
        print("Momentum: {}".format(momentum))

        loss = "categorical_crossentropy"
        print("Loss: {}".format(loss))

        metrics = [tf.keras.metrics.CategoricalAccuracy(name="Acc", dtype=None), Precision, Recall, F1Score, ]

        model.compile(optimizer=opt, loss=loss, metrics=metrics, run_eagerly=True)

        results_dir = base_path + "results/"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        evaluation_dir = base_path + "evaluation/" + type_model
        if not os.path.isdir(evaluation_dir):
            os.makedirs(evaluation_dir)

        # Train the model
        # time, checkpoint = training.training_model(model, train_generator, val_generator, results_prefix)

        # Save the model
        # model.save(type_model + '_model.h5')

        # Load weights in case of needing to validate the trained model again
        model.load_weights('weights/Conv3DLSTM_V6_GaitEvents_BS32_bestw.h5')
        print(type_model + ' model weights loaded.')

        # Get Predictions and Ground Truths
        """print('Predicting in the validation set...')
        prediction = model.predict(val_generator, verbose=1)

        val_true = []
        val_pred = np.argmax(prediction, axis=1)

        for batch in range(len(val_generator)):
            x_batch, y_batch = (val_generator[batch][0], val_generator[batch][1])
            val_label = y_batch.argmax(axis=1)
            for pos in range(len(val_label)):
                val_true.append(val_label[pos])
        
        if plot_val_true_pred:
            save_plot_true_pred(participants=VAL_PARTICIPANTS,
                                generator=val_generator,
                                data='val',
                                pred_labels=val_pred,
                                true_labels=val_true,
                                model=type_model,
                                base_path=base_path)"""

        print('Predicting in the test set...')
        prediction = model.predict(test_generator, verbose=1)

        test_true = []
        test_pred = np.argmax(prediction, axis=1)

        for batch in range(len(test_generator)):
            x_batch, y_batch = (test_generator[batch][0], test_generator[batch][1])
            test_label = y_batch.argmax(axis=1)
            for pos in range(len(test_label)):
                test_true.append(test_label[pos])

        # Get True vs Predicted labels plot
        if plot_test_true_pred:            
            save_plot_true_pred(participants=TEST_PARTICIPANTS,
                                generator=test_generator,
                                data='test',
                                pred_labels=test_pred,
                                true_labels=test_true,
                                model=type_model,
                                base_path=base_path)
        
        """training.call_metrics(y_true=val_true, 
                              y_pred=val_pred, 
                              prediction=prediction, 
                              dir=evaluation_dir, 
                              data='val')
        
        training.call_metrics(y_true=test_true, 
                              y_pred=test_pred, 
                              prediction=prediction, 
                              dir=evaluation_dir,
                              data='test')"""

        # print('Time of training: {} seconds'.format(time))
        # print('Network weights saved in: {}'.format(checkpoint))


