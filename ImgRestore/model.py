import keras


def single_model(input_shape=(None, None, 6), kernel_size=5):
    input = keras.layers.Input(shape=input_shape)
    output = keras.layers.Conv2D(filters=3, kernel_size=kernel_size, padding='same', activation='relu', name='conv1')(
        input)
    model = keras.models.Model(inputs=input, outputs=output)
    return model


def deep_model(input_shape=(None, None, 6)):
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1')(input)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    x1 = x

    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x2 = x

    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, activation='relu')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)

    x = keras.layers.Add()([x, x2])
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)

    x = keras.layers.Add()([x, x1])

    output = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu')(x)

    model = keras.models.Model(inputs=input, outputs=output)

    return model


if __name__ == '__main__':
    model = deep_model(input_shape=(512, 512, 6))
    model.summary()
