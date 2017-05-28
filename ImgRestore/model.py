import keras
import keras.backend.tensorflow_backend as ktf
from keras.layers import *
import tensorflow as tf

def loss2acc(y_true, y_pred,train=False):
    y_true=tf.to_float(y_true)
    y_pred=tf.to_float(y_pred)
    if train :#or  ktf.int_shape(y_true)==ktf.int_shape(y_pred):
        tt= K.mean(K.square(y_pred - y_true), axis=-1) # when training we just compare the output 3chnls and gt 3chnls
    else:
        tt = my_mse(y_true, y_pred) # when finetune we use compare between 6 chnls and 3 chnls
    return  -tt # -10. * K.log(tt) / K.log(K.cast_to_floatx(10.))


def my_mse(y_true, y_pred):
    # print type(y_true), ktf.int_shape(y_true), ktf.int_shape(y_true)
    y_true = tf.to_float(y_true)
    y_pred = tf.to_float(y_pred)

    mask = y_true[..., 3:]
    y_tt = y_true[..., :3]

    y_pred = K.prod(
        K.stack((y_pred, mask), axis=0),
        axis=0
    )
    return K.mean(K.square(y_pred - y_tt), axis=-1)


def padding(x):
    if ktf.int_shape(input)[1] % 2 != 0:
        x = ZeroPadding2D(padding=((1, 0), (0, 0)))(x)
    elif ktf.int_shape(input)[2] % 2 != 0:
        x = ZeroPadding2D(padding=((0, 0), (1, 0)))(x)
    print ktf.int_shape(input)
    return x


def single_model(input_shape=(None, None, 6), n1=5):
    input = Input(shape=input_shape)
    padding='same'
    x=Conv2D(filters=64, kernel_size=n1, padding=padding, activation='relu', name='conv0')(input)
    output = Conv2D(filters=3, kernel_size=n1, padding=padding, activation='relu', name='conv1')(x)

    model = keras.models.Model(inputs=input, outputs=output)

    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=[my_mse], metrics=[loss2acc])
    return model


def deep_model(input_shape=(None, None, 6),n1=16):
    input = Input(shape=input_shape)
    x = input
    padding = 'same'
    x = Conv2D(filters=n1, kernel_size=3, padding=padding, activation='relu', name='conv1')(x)
    x = Conv2D(filters=n1, kernel_size=3, padding=padding, activation='relu', name='conv2')(x)
    x1 = x

    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=n1 * 2, kernel_size=3, padding=padding, activation='relu')(x)
    x2 = x

    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=n1 * 4, kernel_size=3, padding=padding, activation='relu')(x)
    x = Conv2DTranspose(filters=n1 * 4, kernel_size=2, strides=2, padding=padding, activation='relu')(
        x)
    x = Conv2D(filters=n1 * 4, kernel_size=3, padding=padding, activation='relu')(x)
    x = Conv2D(filters=n1 * 2, kernel_size=3, padding=padding, activation='relu')(x)

    assert ktf.int_shape(x)[-1] == ktf.int_shape(x2)[-1]
    x = Add()([x, x2])
    x = Conv2DTranspose(filters=n1 * 2, kernel_size=2, strides=2, padding=padding, activation='relu')(
        x)
    x = Conv2D(filters=n1, kernel_size=3, padding=padding, activation='relu')(x)
    x = Conv2D(filters=n1, kernel_size=3, padding=padding, activation='relu')(x)
    assert ktf.int_shape(x)[-1] == ktf.int_shape(x1)[-1]
    x = Add()([x, x1])

    output = Conv2D(filters=3, kernel_size=3, padding=padding, activation='relu')(x)

    model = keras.models.Model(inputs=input, outputs=output)

    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=[my_mse], metrics=[loss2acc])
    return model


def denoise_model(input_shape=(8, 8, 3), n1=16):
    init = Input(shape=input_shape)
    level1_1 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(init)
    level2_1 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(level1_1)

    level2_2 = Convolution2DTranspose(n1, (3, 3), activation='relu', padding='same')(level2_1)
    level2 = Add()([level2_1, level2_2])

    level1_2 = Convolution2DTranspose(n1, (3, 3), activation='relu', padding='same')(level2)
    level1 = Add()([level1_1, level1_2])

    decoded = Convolution2D(3, (5, 5), activation='linear', padding='same')(level1)

    model = keras.models.Model(init, decoded)
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=[my_mse], metrics=[loss2acc])

    return model


def deep_denoise_model(input_shape=(8, 8, 3), n1=16, n2=32, n3=64):
    init = Input(shape=input_shape)
    c1 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(init)
    c1 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Convolution2D(n2, (3, 3), activation='relu', padding='same')(x)
    c2 = Convolution2D(n2, (3, 3), activation='relu', padding='same')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Convolution2D(n3, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D()(c3)

    c2_2 = Convolution2D(n2, (3, 3), activation='relu', padding='same')(x)
    c2_2 = Convolution2D(n2, (3, 3), activation='relu', padding='same')(c2_2)

    m1 = Add()([c2, c2_2])
    m1 = UpSampling2D()(m1)

    c1_2 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(m1)
    c1_2 = Convolution2D(n1, (3, 3), activation='relu', padding='same')(c1_2)

    m2 = Add()([c1, c1_2])

    decoded = Convolution2D(3, (5, 5), activation='relu', padding='same')(m2)

    model = keras.models.Model(init, decoded)
    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss=[my_mse], metrics=[loss2acc])

    return model



if __name__ == '__main__':
    # model = deep_model(input_shape=(512, 512, 6))
    model = deep_denoise_model()
    model.summary()
    single_model().summary()
