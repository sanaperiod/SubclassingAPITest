import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from util import ISubclassingSummary


class MyNet(Model, ISubclassingSummary):
    def __init__(self):
        super().__init__()
        self.dense_0 = Dense(units=30, activation="sigmoid", name="dense0")
        self.dense_1 = Dense(units=20, activation="sigmoid", name="dense1")
        self.dense_2 = Dense(units=10, activation="sigmoid", name="dense2")

    def call(self, inputs):
        x = self.dense_0(inputs)
        x = self.dense_1(x)
        outputs = self.dense_2(x)
        return outputs


class MyBlock(Layer):
    def __init__(self):
        super().__init__()
        self.dense_0 = Dense(units=300, activation="sigmoid")
        self.dense_1 = Dense(units=200, activation="sigmoid")
        self.dense_2 = Dense(units=100, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_0(inputs)
        x = self.dense_1(x)
        outputs = self.dense_2(x)
        return outputs


class MyBlockNet(Model, ISubclassingSummary):
    def __init__(self):
        super().__init__()
        self.dense_0 = Dense(units=20, activation="sigmoid", name="outer_dense0")
        self.block_0 = MyBlock()
        self.block_1 = MyBlock()
        self.dense_1 = Dense(units=10, activation="sigmoid", name="outer_dense1")

    def call(self, inputs):
        x = self.dense_0(inputs)
        x = self.block_0(x)
        x = self.block_1(x)
        outputs = self.dense_1(x)
        return outputs


class MyResNet(Model, ISubclassingSummary):
    def __init__(self):
        super().__init__()
        self.dense_0 = Dense(units=30, activation="sigmoid", name="dense0")
        self.dense_1 = Dense(units=20, activation="sigmoid", name="dense1")
        self.dense_2 = Dense(units=10, activation="sigmoid", name="dense2")

    def call(self, inputs, training=None):
        x = self.dense_0(inputs)
        print(training)
        if tf.random.uniform((1,)) > 0.5 or not training:
            x = self.dense_1(x)
        outputs = self.dense_2(x)
        return outputs


def simple_model_test(input_shape):
    # test1
    subclassing_model_a = MyNet()
    try:
        subclassing_model_a.summary()
    except Exception as e:
        print(repr(e))
    # test2
    subclassing_model_b = MyNet()
    # noinspection PyCallingNonCallable
    subclassing_model_b(tf.zeros(input_shape))
    subclassing_model_b.summary()
    # test3
    subclassing_model_c = MyNet()
    subclassing_model_c.my_summary(input_shape)


def main():
    INPUT_SHAPE = (28, 28)
    # MyNet test
    simple_model_test(INPUT_SHAPE)
    # MyBlockNet test
    MyBlockNet().my_summary(INPUT_SHAPE)
    # MyResNet test
    MyResNet().my_summary(INPUT_SHAPE)


if __name__ == '__main__':
    main()
