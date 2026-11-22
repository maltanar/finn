# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class BitGlue(HWCustomOp):
    """Zero-LUT streaming bit-level Gather/Unpack/Concat/Const-Tieoff layer.

    Consumes 1 or 2 fully-parallel streaming inputs and produces 1 output stream
    where each output bit is wired from a specified (stream, bit_index) or
    constant 0/1.
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "NumIn0Elems": ("i", True, 0),
            "NumIn1Elems": ("i", False, 0),
            "NumOutElems": ("i", True, 0),
            "Input0Type": ("s", True, ""),
            "Input1Type": ("s", False, ""),
            "OutputType": ("s", True, ""),
            "numInputVectors": ("ints", False, [1]),
            # Per output bit (length NumOutElems * out_bw):
            #   SrcStream[j] in {0: in0, 1: in1, -1: const 0, -2: const 1}
            #   SrcBit[j] = bit index in selected input stream
            "SrcStream": ("ints", True, []),
            "SrcBit": ("ints", True, []),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_num_inputs(self):
        return 2 if self.get_nodeattr("NumIn1Elems") > 0 else 1

    def get_normal_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        if ind == 0:
            return tuple(vecs + [self.get_nodeattr("NumIn0Elems")])
        elif ind == 1:
            return tuple(vecs + [self.get_nodeattr("NumIn1Elems")])
        raise Exception(f"Undefined input ind {ind} for BitGlue")

    def get_normal_output_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [self.get_nodeattr("NumOutElems")])

    def get_folded_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        elems = self.get_nodeattr("NumIn0Elems" if ind == 0 else "NumIn1Elems")
        return tuple(vecs + [1, elems])

    def get_folded_output_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [1, self.get_nodeattr("NumOutElems")])

    def get_input_datatype(self, ind=0):
        if ind == 0:
            return DataType[self.get_nodeattr("Input0Type")]
        elif ind == 1:
            return DataType[self.get_nodeattr("Input1Type")]
        raise Exception(f"Undefined input ind {ind} for BitGlue")

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("OutputType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt0 = model.get_tensor_datatype(node.input[0])
        if idt0 != self.get_input_datatype(0):
            self.set_nodeattr("Input0Type", idt0.name)
        if len(node.input) > 1:
            idt1 = model.get_tensor_datatype(node.input[1])
            if idt1 != self.get_input_datatype(1):
                self.set_nodeattr("Input1Type", idt1.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_instream_width(self, ind=0):
        elems = self.get_nodeattr("NumIn0Elems" if ind == 0 else "NumIn1Elems")
        return elems * self.get_input_datatype(ind).bitwidth()

    def get_outstream_width(self, ind=0):
        return self.get_nodeattr("NumOutElems") * self.get_output_datatype().bitwidth()

    def get_number_output_values(self):
        return int(np.prod(self.get_nodeattr("numInputVectors")))

    def get_exp_cycles(self):
        return self.get_number_output_values()

    def _unpack_bits(self, arr, bw):
        # arr: (N, E) -> (N, E * bw) LSB-first per element
        arr_i = arr.reshape(-1, arr.shape[-1]).astype(np.int64)
        if bw == 1:
            return (arr_i & 1).astype(np.uint8)
        bits = [(arr_i >> b) & 1 for b in range(bw)]
        # interleave: element e bit b at e*bw + b
        stacked = np.stack(bits, axis=-1)  # (N, E, bw)
        return stacked.reshape(arr_i.shape[0], arr_i.shape[1] * bw).astype(np.uint8)

    def _pack_bits(self, bits, elems, bw, out_shape):
        # bits: (N, elems * bw) -> out_shape
        if bw == 1:
            return bits.reshape(out_shape).astype(np.float32)
        reshaped = bits.reshape(bits.shape[0], elems, bw).astype(np.int64)
        weights = (1 << np.arange(bw, dtype=np.int64))[None, None, :]
        vals = (reshaped * weights).sum(axis=-1)
        return vals.reshape(out_shape).astype(np.float32)

    def execute_node(self, context, graph):
        node = self.onnx_node
        in0 = context[node.input[0]]
        b0 = self._unpack_bits(in0, self.get_input_datatype(0).bitwidth())
        n_vecs = b0.shape[0]

        src_stream = np.asarray(self.get_nodeattr("SrcStream"), dtype=np.int64)
        src_bit = np.asarray(self.get_nodeattr("SrcBit"), dtype=np.int64)
        out_bits_total = self.get_outstream_width(0)
        assert len(src_stream) == out_bits_total and len(src_bit) == out_bits_total

        out_bits = np.zeros((n_vecs, out_bits_total), dtype=np.uint8)
        m0 = src_stream == 0
        if np.any(m0):
            out_bits[:, m0] = b0[:, src_bit[m0]]
        if len(node.input) > 1:
            in1 = context[node.input[1]]
            b1 = self._unpack_bits(in1, self.get_input_datatype(1).bitwidth())
            m1 = src_stream == 1
            if np.any(m1):
                out_bits[:, m1] = b1[:, src_bit[m1]]
        mc1 = src_stream == -2
        if np.any(mc1):
            out_bits[:, mc1] = 1

        out_elems = self.get_nodeattr("NumOutElems")
        out_bw = self.get_output_datatype().bitwidth()
        out_shape = self.get_normal_output_shape()
        context[node.output[0]] = self._pack_bits(out_bits, out_elems, out_bw, out_shape)

    def verify_node(self):
        return []
