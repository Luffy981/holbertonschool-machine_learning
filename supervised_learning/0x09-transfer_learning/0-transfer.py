#!/usr/bin/env python3
"""
Transfer learning...
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Data preprocessing
    """
    # Preprocesses a tensor or Numpy array encoding a batch of images.
    X_p = K.applications.resnet50.preprocess_input(x=X)
    # Converts a class vector (integers) to binary class matrix
    Y_p = K.utils.to_categorical(y=Y)
    return X_p, Y_p


if __name__ == '__main__':
    """
    making transfer learning
    """
    # Loads the CIFAR10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    # Load weights pre-trained on ImageNet.
    # Do not include the ImageNet classifier at the top.
    base_model = K.applications.InceptionResNetV2(weights="imagenet",
                                                  input_shape=(288, 288, 3),
                                                  include_top=False)
    base_model.summary()
    # Preprocessing data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Freeze the base_model
    # base_model.trainable = False

    # Create new model on top
    inputs = K.Input(shape=(32, 32, 3))
    # Resizes the images contained in a 4D tensor.
    input_image = K.layers.Lambda(
    lambda image: K.backend.resize_images(
        x=image,
        height_factor=288/32,
        width_factor=288/32,
        data_format='channels_last'
        )
        )(inputs)
    # Add to base model
    layer = base_model(input_image, training=False)
    # Global average pooling operation for spatial data.
    poolG1 = K.layers.GlobalAveragePooling2D()(layer)

    # stack new 3 layer network
    fc1 = K.layers.Dense(units=512)(poolG1)
    batchN1 = K.layers.BatchNormalization()(fc1)
    activation1 = K.layers.Activation('relu')(batchN1)
    dropout1 = K.layers.Dropout(0.3)(activation1)

    fc2 = K.layers.Dense(units=512)(dropout1)
    batchN2 = K.layers.BatchNormalization()(fc2)
    activation2 = K.layers.Activation('relu')(batchN2)
    dropout2 = K.layers.Dropout(0.3)(activation2)

    fc3 = K.layers.Dense(10, activation='softmax')(dropout2)

    model = K.Model(inputs=inputs, outputs=fc3)

    # Freeze the layers model
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False
    # Compile the model should be done *after* setting layers to non-trainable
    # optimizer
    optimizer = K.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=Y_train,
              batch_size=300,
              shuffle=True,
              epochs=2,
              validation_data=(X_test, Y_test),
              verbose=True)

    # Unfreeze selected layers from IRNV2
    for layer in base_model.layers[:498]:
        layer.trainable = False
    for layer in base_model.layers[498:]:
        layer.trainable = True

    print("\nTraining bottom couple hundred layers\n")
    # optimizer with low learning rate
    optimizer = K.optimizers.Adam(1e-5)
    # Training the top layer
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    history = model.fit(x=X_train,
                        y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=300,
                        epochs=5,
                        verbose=True)
    model.save('cifar10.h5')
    print("saved to ./cifar10.h5 NEVER STOP SMILING! ")
#!/usr/bin/env python3
"""
Transfer learning...
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Data preprocessing
    """
    # Preprocesses a tensor or Numpy array encoding a batch of images.
    X_p = K.applications.resnet50.preprocess_input(x=X)
    # Converts a class vector (integers) to binary class matrix
    Y_p = K.utils.to_categorical(y=Y)
    return X_p, Y_p


if __name__ == '__main__':
    """
    making transfer learning
    """
    # Loads the CIFAR10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    # Load weights pre-trained on ImageNet.
    # Do not include the ImageNet classifier at the top.
    base_model = K.applications.InceptionResNetV2(weights="imagenet",
                                                  input_shape=(299, 299, 3),
                                                  include_top=False)
    base_model.summary()
    # Preprocessing data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Freeze the base_model
    # base_model.trainable = False

    # Create new model on top
    inputs = K.Input(shape=(32, 32, 3))
    # Resizes the images contained in a 4D tensor.
    input_image = K.layers.Lambda(
    lambda image: K.backend.resize_images(
        x=image,
        height_factor=299/32,
        width_factor=299/32,
        data_format='channels_last'
        )
        )(inputs)
    # Add to base model
    layer = base_model(input_image, training=False)
    # Global average pooling operation for spatial data.
    poolG1 = K.layers.GlobalAveragePooling2D()(layer)

    # stack new 3 layer network
    fc1 = K.layers.Dense(units=512)(poolG1)
    batchN1 = K.layers.BatchNormalization()(fc1)
    activation1 = K.layers.Activation('relu')(batchN1)
    dropout1 = K.layers.Dropout(0.3)(activation1)

    fc2 = K.layers.Dense(units=512)(dropout1)
    batchN2 = K.layers.BatchNormalization()(fc2)
    activation2 = K.layers.Activation('relu')(batchN2)
    dropout2 = K.layers.Dropout(0.3)(activation2)

    fc3 = K.layers.Dense(10, activation='softmax')(dropout2)

    model = K.Model(inputs=inputs, outputs=fc3)

    # Freeze the layers model
    base_model.trainable = False
    # Compile the model should be done *after* setting layers to non-trainable
    # optimizer
    optimizer = K.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=Y_train,
              batch_size=300,
              shuffle=True,
              epochs=2,
              validation_data=(X_test, Y_test),
              verbose=True)

    # Unfreeze selected layers from IRNV2
    for layer in base_model.layers[:498]:
        layer.trainable = False
    for layer in base_model.layers[498:]:
        layer.trainable = True

    print("\nTraining bottom couple hundred layers\n")
    # optimizer with low learning rate
    optimizer = K.optimizers.Adam(1e-5)
    # Training the top layer
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    history = model.fit(x=X_train,
                        y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=300,
                        epochs=5,
                        verbose=True)
    model.save('cifar10.h5')
    print("saved to ./cifar10.h5 NEVER STOP SMILING! ")
