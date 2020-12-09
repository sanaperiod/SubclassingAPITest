from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense


def simple_net_test(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(units=30, activation="sigmoid", name="dense0")(inputs)
    x = Dense(units=20, activation="sigmoid", name="dense1")(x)
    outputs = Dense(units=10, activation="sigmoid", name="dense2")(x)
    func_model = Model(inputs=inputs, outputs=outputs)
    func_model.summary()


def my_block(inputs):
    x = Dense(units=300, activation="sigmoid")(inputs)
    x = Dense(units=200, activation="sigmoid")(x)
    outputs = Dense(units=100, activation="sigmoid")(x)
    return outputs


def block_net_test(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(units=20, activation="sigmoid")(inputs)
    x = my_block(x)
    x = my_block(x)
    outputs = Dense(units=10, activation="sigmoid")(x)
    Model(inputs=inputs, outputs=outputs, name="f_block_net").summary()


def main():
    INPUT_SHAPE = (28, 28)
    simple_net_test(INPUT_SHAPE)

    block_net_test(INPUT_SHAPE)


if __name__ == '__main__':
    main()
