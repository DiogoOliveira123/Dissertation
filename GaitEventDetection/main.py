from create_dataset import *
from models import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import CategoricalAccuracy
from utils import *
import tensorflow as tf
import sklearn.metrics as skm

if __name__ == '__main__':

    random.seed(0)
    state = random.getstate()

    base_path = "/home/birdlab/Desktop/WALKIT_SW/"
    dataset_path = "/home/birdlab/Desktop/WALKIT_SW/dataset/dataset_gaitevents"

    NUM_PARTICIPANTS = 15
    TEST_PARTICIPANTS = ['15', '08', '03']
    VAL_PARTICIPANTS = ['02', '11']

    train_network = True

    if not train_network:
        # balanced_dataset, total_num_labels = CreateBalancedDataset()
        data = Dataset()
        train, val, test = data.SplitDataset(NUM_PARTICIPANTS, TEST_PARTICIPANTS, VAL_PARTICIPANTS)

    else:
        # train parameters
        epochs = 200
        batch_size = 64
        num_classes = 3
        type_model = "ResNet50"

        ## models
        results_prefix = type_model + "_GaitEvents_BS" + str(batch_size)
        print("\nNeural network: {}".format(results_prefix))

        print("Epochs: {}".format(epochs))
        print("Batch Size: {}".format(batch_size))

        checkpoint_metric = "val_F1Score"

        training = TrainModel(checkpoint_metric_name=checkpoint_metric, no_epochs=epochs, base_path=base_path,
                              dataset_path=dataset_path, num_classes=num_classes,
                              fixed_width=224, batch_size=batch_size)

        train_dir = os.path.join(dataset_path, 'treino')
        validation_dir = os.path.join(dataset_path, 'val')
        test_dir = os.path.join(dataset_path, 'teste')

        train_generator, val_generator, test_generator = training.dataGenerator(train_dir, validation_dir, test_dir,
                                                                                val_shuffle=True, val_augment=False)

        model = training.build_model(type_model=type_model, weights="imagenet", input_shape=(224, 224, 3))

        model.summary()

        learning_rate = 0.0001
        decay = 0.005
        momentum = 0.9
        nesterov = True

        opt = tf.keras.optimizers.legacy.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            name='SGD'
        )

        print("Learning rate: {}".format(learning_rate))
        print("Decay: {}".format(decay))
        print("Momentum: {}".format(momentum))

        loss = "categorical_crossentropy"
        print("Loss: {}".format(loss))

        metrics = [CategoricalAccuracy(name="Acc", dtype=None), Precision, Recall, F1Score]

        model.compile(optimizer=opt, loss=loss, metrics=metrics, run_eagerly=True)

        time, checkpoint = training.training_model(model, train_generator, val_generator, results_prefix)

        # Load neural network weights in checkpoint
        model.load_weights(checkpoint)

        # Get Predictions and Ground Truths
        print('Predicting in the validation set...')
        _, val_generator, _ = training.dataGenerator(train_dir, validation_dir, test_dir, False, False)

        prediction = model.predict(val_generator, verbose=1)

        val_pred = np.argmax(prediction, axis=1)
        val_true = []

        for batch in range(len(val_generator)):
            x_batch, y_batch = (val_generator[batch][0], val_generator[batch][1])
            val_label = y_batch.argmax(axis=1)
            for pos in range(len(val_label)):
                val_true.append(val_label[pos])

        ConfMat = skm.confusion_matrix(val_true, val_pred)
        np.savetxt(base_path + "results/" + results_prefix + "_val_confusionMatrix.csv", ConfMat, delimiter=",")

        print('Predicting in the test set...')
        prediction = model.predict(test_generator, verbose=1)

        y_pred = np.argmax(prediction, axis=1)

        ConfMat = skm.confusion_matrix(test_generator.labels, y_pred)
        np.savetxt(base_path + "results/" + results_prefix + "_test_confusionMatrix.csv", ConfMat, delimiter=",")
