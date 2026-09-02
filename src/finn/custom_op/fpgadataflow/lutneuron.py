# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.lnn.lookup_table import lookup_table

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class LUTNeuron(HWCustomOp):
    """Abstraction layer for HW implementation of the qonnx.custom_op.lnn
    LookupTable operator: a layer of M neurons, each of which reads K elements
    from the last axis of the input, concatenates them into an address and
    emits the corresponding entry of its own truth table.

    The `indices` ([M, K]) and `table` ([M, 2**(K*input_bits)]) node inputs are
    compile-time constants: they are baked into the generated hardware and do
    not become streaming interfaces."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of elements along the last axis of the input (C_in)
            "NumInputs": ("i", True, 0),
            # number of neurons in this layer (M)
            "NumNeurons": ("i", True, 0),
            # fan-in of every neuron (K)
            "FanIn": ("i", True, 0),
            # datatype of the input elements, bitwidth == input_bits
            "InputType": ("s", True, ""),
            # datatype of the output elements, bitwidth == effective out_bits
            "OutputType": ("s", True, ""),
            # leading (batch/spatial) axes of the input tensor
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_bits(self):
        return self.get_input_datatype().bitwidth()

    def get_out_bits(self):
        return self.get_output_datatype().bitwidth()

    def get_table_size(self):
        """Number of entries per neuron, S = 2**(K * input_bits)."""
        return 2 ** (self.get_nodeattr("FanIn") * self.get_input_bits())

    def get_normal_input_shape(self, ind=0):
        if ind == 0:
            return tuple(
                list(self.get_nodeattr("numInputVectors")) + [self.get_nodeattr("NumInputs")]
            )
        elif ind == 1:
            return (self.get_nodeattr("NumNeurons"), self.get_nodeattr("FanIn"))
        elif ind == 2:
            return (self.get_nodeattr("NumNeurons"), self.get_table_size())
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_normal_output_shape(self, ind=0):
        return tuple(list(self.get_nodeattr("numInputVectors")) + [self.get_nodeattr("NumNeurons")])

    def get_folded_input_shape(self, ind=0):
        # fully parallel: the whole C_in axis is consumed in a single beat
        if ind == 0:
            return tuple(
                list(self.get_nodeattr("numInputVectors")) + [1, self.get_nodeattr("NumInputs")]
            )
        else:
            return self.get_normal_input_shape(ind)

    def get_folded_output_shape(self, ind=0):
        return tuple(
            list(self.get_nodeattr("numInputVectors")) + [1, self.get_nodeattr("NumNeurons")]
        )

    def get_input_datatype(self, ind=0):
        if ind == 0:
            return DataType[self.get_nodeattr("InputType")]
        elif ind in [1, 2]:
            # indices/table are compile-time constants, not streams
            return DataType[self.get_nodeattr("OutputType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("OutputType")]

    def get_instream_width(self, ind=0):
        if ind == 0:
            return self.get_input_bits() * self.get_nodeattr("NumInputs")
        elif ind in [1, 2]:
            return 0
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_outstream_width(self, ind=0):
        return self.get_out_bits() * self.get_nodeattr("NumNeurons")

    def get_number_output_values(self):
        return int(np.prod(self.get_nodeattr("numInputVectors")))

    def get_exp_cycles(self):
        return 0

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "InputType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
            self.set_nodeattr("InputType", idt.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def verify_node(self):
        info_messages = []
        try:
            for attr in ["NumInputs", "NumNeurons", "FanIn", "InputType", "OutputType"]:
                self.get_nodeattr(attr)
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """LUTNeuron needs the following attributes:
                NumInputs, NumNeurons, FanIn, InputType, OutputType"""
            )
        if len(self.onnx_node.input) == 3:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("LUTNeuron needs 3 inputs (X, indices, table)")
        return info_messages

    def execute_node(self, context, graph):
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        x = context[node.input[0]]
        assert tuple(x.shape) == exp_ishape, "Input shape doesn't match expected shape."
        indices = np.asarray(context[node.input[1]]).astype(np.int64)
        table = context[node.input[2]]
        y = lookup_table(x, indices, table, input_bits=self.get_input_bits())
        context[node.output[0]] = y.astype(np.float32).reshape(exp_oshape)

    def lut_estimation(self, fpgapart):
        # each neuron maps onto ceil(S/64) LUT6 per output bit
        n_lut6 = int(np.ceil(self.get_table_size() / 64))
        return self.get_nodeattr("NumNeurons") * self.get_out_bits() * n_lut6

    def bram_estimation(self, fpgapart):
        return 0

    def dsp_estimation(self, fpgapart):
        return 0
