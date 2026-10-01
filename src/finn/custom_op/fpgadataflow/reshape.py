"""Hardware operator corresponding to the standard ONNX Reshape."""

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


@register_custom_op
class Reshape(HWCustomOp):
    """Reshape operator implemented as an AXI pass-through data stream."""

    def get_nodeattr_types(self):
        attrs = HWCustomOp.get_nodeattr_types(self)
        attrs.update(
            {
                "inp_shape": ("ints", True, [1]),
                "out_shape": ("ints", True, [1]),
                "dtype": ("s", True, ""),
                "PE": ("i", False, 1),
            }
        )
        return attrs

    @property
    def inp_shape(self):
        return self.get_nodeattr("inp_shape")

    @property
    def out_shape(self):
        return self.get_nodeattr("out_shape")

    @property
    def dtype(self):
        return DataType[self.get_nodeattr("dtype")]

    @property
    def pe(self):
        return self.get_nodeattr("PE")

    def get_input_datatype(self, ind=0):
        return self.dtype

    def get_output_datatype(self, ind=0):
        return self.dtype

    def get_normal_input_shape(self, ind=0):
        return self.inp_shape

    def get_normal_output_shape(self, ind=0):
        return self.out_shape

    def get_folded_input_shape(self, ind=0):
        *num_inputs, num_elems = self.inp_shape
        assert num_elems % self.pe == 0, "PE must divide last axis"
        return *num_inputs, num_elems // self.pe, self.pe

    def get_folded_output_shape(self, ind=0):
        *num_outputs, num_elems = self.out_shape
        assert num_elems % self.pe == 0, "PE must divide last axis"
        return *num_outputs, num_elems // self.pe, self.pe

    def get_instream_width(self, ind=0):
        return self.get_folded_input_shape()[-1] * self.dtype.bitwidth()

    def get_outstream_width(self, ind=0):
        return self.get_folded_output_shape()[-1] * self.dtype.bitwidth()

    def get_number_output_values(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self):
        return self.get_number_output_values()

    def infer_node_datatype(self, model: ModelWrapper):
        node = self.onnx_node
        input_dtype = model.get_tensor_datatype(node.input[0])
        if input_dtype != self.dtype:
            self.set_nodeattr("dtype", input_dtype.name)
        model.set_tensor_datatype(node.output[0], self.dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        context[node.output[0]] = np.reshape(
            context[node.input[0]], newshape=self.out_shape
        ).astype(np.float32)
