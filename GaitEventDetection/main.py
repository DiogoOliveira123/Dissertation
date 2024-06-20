from create_dataset import *
from sequence_generator import *
from models import TrainModel
import tensorflow as tf
import sklearn.metrics as skm
from utils import Precision, Recall, F1Score

if __name__ == '__main__':

    random.seed(0)
    state = random.getstate()

    base_path = '/home/birdlab/Desktop/WALKIT_SW/code/GaitEventDetection/'
    dataset_path = '/home/birdlab/Desktop/WALKIT_SW/dataset/RGB_labeling_30Hz_balanced_aligned_v6.xlsx'

    NUM_PARTICIPANTS = 15
    TEST_PARTICIPANTS = ['15', '8', '3']
    VAL_PARTICIPANTS = ['2', '11']

    train_network = True

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

        type_model = "Conv3DLSTM_V3"

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

        # Split TRAIN, VAL, TEST info from CSV FILE
        train_df, val_df, test_df = SequenceDataGenerator.splitTrainValTest(dataset_path)

        # SEQUENCE GENERATOR
        train_generator = SequenceDataGenerator(train_df, batch_size, augment=True)
        val_generator = SequenceDataGenerator(val_df, batch_size, augment=False)
        test_generator = SequenceDataGenerator(test_df, batch_size, augment=False)

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

        evaluation_dir = base_path + "evaluation/"
        if not os.path.isdir(evaluation_dir):
            os.makedirs(evaluation_dir)

        # Train the model
        time, checkpoint = training.training_model(model, train_generator, val_generator, results_prefix)

        # Save the model
        model.save('conv3dlstm_model.h5')

        # Get Predictions and Ground Truths
        print('Predicting in the validation set...')
        prediction = model.predict(val_generator, verbose=1)

        val_true = []
        val_pred = np.argmax(prediction, axis=1)

        for batch in range(len(val_generator)):
            x_batch, y_batch = (val_generator[batch][0], val_generator[batch][1])
            val_label = y_batch.argmax(axis=1)
            for pos in range(len(val_label)):
                val_true.append(val_label[pos])

        training.call_metrics(y_true=val_true, y_pred=val_pred, prediction=prediction, dir=evaluation_dir, data='val')

        print('Predicting in the test set...')
        prediction = model.predict(test_generator, verbose=1)

        test_true = []
        test_pred = np.argmax(prediction, axis=1)

        for batch in range(len(test_generator)):
            x_batch, y_batch = (test_generator[batch][0], test_generator[batch][1])
            test_label = y_batch.argmax(axis=1)
            for pos in range(len(test_label)):
                test_true.append(test_label[pos])

        training.call_metrics(y_true=test_true, y_pred=test_pred, prediction=prediction, dir=evaluation_dir, data='test')

        print('Time of training: {} seconds'.format(time))
        print('Network weights saved in: {}'.format(checkpoint))


