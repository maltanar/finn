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

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.lnn.lookup_table import lookup_table
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferLUTNeuronLayer
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 5

# (input_bits, fan_in, table onnx elem type, out_bits)
CONFIGS = [
    (1, 2, TensorProto.UINT8, 1),  # two-input logic gates
    (1, 6, TensorProto.UINT8, 0),  # 6:8 LUT, full uint8 output
    (2, 3, TensorProto.UINT8, 4),  # multi-bit input and output
    (1, 4, TensorProto.INT8, 0),  # signed table
]


def _out_datatype(elem_type, out_bits):
    full_bits, signed = {TensorProto.UINT8: (8, False), TensorProto.INT8: (8, True)}[elem_type]
    bits = out_bits if out_bits > 0 else full_bits
    if bits == 1 and not signed:
        return DataType["BINARY"]
    return DataType["%s%d" % ("INT" if signed else "UINT", bits)]


def make_lookuptable_model(c_in, m, k, input_bits, elem_type, out_bits, n_vecs=1):
    """Builds a single-node qonnx.custom_op.lnn LookupTable model with random
    indices and table."""
    rng = np.random.default_rng(0)
    s = 2 ** (k * input_bits)
    indices = rng.integers(0, c_in, size=(m, k)).astype(np.int64)
    odt = _out_datatype(elem_type, out_bits)
    np_dtype = np.uint8 if elem_type == TensorProto.UINT8 else np.int8
    table = rng.integers(int(odt.min()), int(odt.max()) + 1, size=(m, s)).astype(np_dtype)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [n_vecs, c_in])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [n_vecs, m])
    node = helper.make_node(
        "LookupTable",
        ["inp", "indices", "table"],
        ["outp"],
        domain="qonnx.custom_op.lnn",
        name="lut0",
        input_bits=input_bits,
        out_bits=out_bits,
    )
    graph = helper.make_graph([node], "lutneuron_graph", [inp], [outp])
    model = ModelWrapper(helper.make_model(graph, producer_name="lutneuron-test"))
    model.set_initializer("indices", indices)
    model.graph.initializer.append(
        helper.make_tensor("table", elem_type, list(table.shape), table.flatten().tolist())
    )
    idt = DataType["BINARY"] if input_bits == 1 else DataType["UINT%d" % input_bits]
    model.set_tensor_datatype("inp", idt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model, indices, table, idt, odt


def _random_input(idt, shape):
    rng = np.random.default_rng(1)
    return rng.integers(0, 2 ** idt.bitwidth(), size=shape).astype(np.float32)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("input_bits, k, elem_type, out_bits", CONFIGS)
def test_fpgadataflow_lutneuron_convert(input_bits, k, elem_type, out_bits):
    """InferLUTNeuronLayer produces a correctly parametrized LUTNeuron node
    whose python execution matches the numpy reference."""
    c_in, m, n_vecs = 16, 8, 3
    model, indices, table, idt, odt = make_lookuptable_model(
        c_in, m, k, input_bits, elem_type, out_bits, n_vecs
    )
    x = _random_input(idt, (n_vecs, c_in))
    golden = lookup_table(x, indices, table, input_bits=input_bits).astype(np.float32)

    model = model.transform(InferLUTNeuronLayer())
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "LUTNeuron"
    inst = getCustomOp(node)
    assert inst.get_nodeattr("NumInputs") == c_in
    assert inst.get_nodeattr("NumNeurons") == m
    assert inst.get_nodeattr("FanIn") == k
    assert inst.get_nodeattr("InputType") == idt.name
    assert inst.get_nodeattr("OutputType") == odt.name
    assert inst.get_nodeattr("numInputVectors") == [n_vecs]
    assert inst.get_normal_input_shape(0) == (n_vecs, c_in)
    assert inst.get_normal_output_shape(0) == (n_vecs, m)
    assert inst.get_instream_width(0) == c_in * input_bits
    assert inst.get_outstream_width(0) == m * odt.bitwidth()
    # indices/table must not turn into streaming interfaces
    assert inst.get_instream_width(1) == 0
    assert inst.get_instream_width(2) == 0

    ret = execute_onnx(model, {"inp": x})["outp"]
    assert (ret == golden).all()


def _make_asymmetric_gate_model():
    """Single neuron computing a0 AND NOT a1 over inputs 3 (slot 0) and 1
    (slot 1); the table is asymmetric so swapping the slot significance would
    change the result."""
    indices = np.array([[3, 1]], dtype=np.int64)
    table = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [4, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [4, 1])
    node = helper.make_node(
        "LookupTable",
        ["inp", "indices", "table"],
        ["outp"],
        domain="qonnx.custom_op.lnn",
        name="lut0",
        input_bits=1,
        out_bits=1,
    )
    graph = helper.make_graph([node], "asym_graph", [inp], [outp])
    model = ModelWrapper(helper.make_model(graph, producer_name="lutneuron-test"))
    model.set_initializer("indices", indices)
    model.graph.initializer.append(
        helper.make_tensor("table", TensorProto.UINT8, [1, 4], table.flatten().tolist())
    )
    model.set_tensor_datatype("inp", DataType["BINARY"])
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    # rows enumerate (a1, a0) = (0,0), (0,1), (1,0), (1,1) on positions 1 and 3
    x = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1]],
        dtype=np.float32,
    )
    expected = np.array([[0], [1], [0], [0]], dtype=np.float32)
    return model, x, expected


@pytest.mark.fpgadataflow
def test_fpgadataflow_lutneuron_slot_order():
    model, x, expected = _make_asymmetric_gate_model()
    assert (execute_onnx(model, {"inp": x})["outp"] == expected).all()
    model = model.transform(InferLUTNeuronLayer())
    assert (execute_onnx(model, {"inp": x})["outp"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_lutneuron_slot_order_rtlsim():
    model, x, expected = _make_asymmetric_gate_model()
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))
    assert (execute_onnx(model, {"inp": x})["outp"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.parametrize("input_bits, k, elem_type, out_bits", CONFIGS)
@pytest.mark.parametrize("input_reg, output_reg", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_fpgadataflow_lutneuron_rtlsim(input_bits, k, elem_type, out_bits, input_reg, output_reg):
    c_in, m, n_vecs = 16, 8, 3
    model, indices, table, idt, _ = make_lookuptable_model(
        c_in, m, k, input_bits, elem_type, out_bits, n_vecs
    )
    x = _random_input(idt, (n_vecs, c_in))
    golden = lookup_table(x, indices, table, input_bits=input_bits).astype(np.float32)

    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(SpecializeLayers(test_fpga_part))
    assert model.graph.node[0].op_type == "LUTNeuron_rtl"
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("input_reg", input_reg)
    inst.set_nodeattr("output_reg", output_reg)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))

    ret = execute_onnx(model, {"inp": x})["outp"]
    assert (ret == golden).all()
