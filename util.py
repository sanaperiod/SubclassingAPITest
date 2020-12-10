from tensorflow.keras import Input, Model


# from abc import abstractmethod
# from typing import Any, Optional

# Can be used with python 3.8 or later
# from typing import final

# Can be used with python 3.9 or later
# from typing import Protocol

class ISubclassingSummary:
    # Can be used with python 3.8 or later
    # @final
    def my_summary(self, input_shape):
        tmp_x = Input(shape=input_shape, name='tmp_input')
        # noinspection PyUnresolvedReferences
        tmp_m = Model(inputs=tmp_x, outputs=self.call(tmp_x), name='tmp_model')
        tmp_m.summary()
        del tmp_x, tmp_m

    # @abstractmethod
    # def call(self, inputs: Any) -> Optional[Any]:
    #     pass

# Can be used with python 3.9 or later
# class ISubclassingSummaryPy39(Protocol):
#     # Can be used with python 3.8 or later
#     # @final
#     def my_summary(self, input_shape):
#         tmp_x = Input(shape=input_shape, name='tmp_input')
#         # noinspection PyUnresolvedReferences
#         tmp_m = Model(inputs=tmp_x, outputs=self.call(tmp_x), name='tmp_model')
#         tmp_m.summary()
#         del tmp_x, tmp_m
#
# def my_summary(input_shape, model:ISubclassingSummaryPy39):
#     tmp_x = Input(shape=input_shape, name='tmp_input')
#     # noinspection PyUnresolvedReferences
#     tmp_m = Model(inputs=tmp_x, outputs=model.call(tmp_x), name='tmp_model')
#     tmp_m.summary()
#     del tmp_x, tmp_m