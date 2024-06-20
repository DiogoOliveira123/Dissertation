from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Flatten, Dense, Dropout, LSTM
from keras.layers import TimeDistributed


def Conv3DLSTM(num_classes, input_shape=(64, 4, 224, 224, 3)):
    # CNN-LSTM network, also known as LRCN
    model = Sequential()

    model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same', input_shape=input_shape[1:]))
    model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
