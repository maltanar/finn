# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Abs(HWCustomOp):
    """Abstraction layer for HW implementation of elementwise integer Abs."""

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 1),
            "InputDataType": ("s", True, ""),
            "OutputDataType": ("s", True, ""),
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        ch = self.get_nodeattr("NumChannels")
        return tuple(vecs + [ch])

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape(ind)

    def get_folded_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        assert ch % pe == 0, "PE must divide NumChannels"
        return tuple(vecs + [ch // pe, pe])

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape(ind)

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("InputDataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("OutputDataType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            self.set_nodeattr("InputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_instream_width(self, ind=0):
        return self.get_nodeattr("PE") * self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind=0):
        return self.get_nodeattr("PE") * self.get_output_datatype().bitwidth()

    def get_number_output_values(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self):
        return self.get_number_output_values()

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp = context[node.input[0]]
        context[node.output[0]] = np.abs(inp).astype(np.float32)

    def verify_node(self):
        return []
