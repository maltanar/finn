import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.absorb_transpose_into_lutneuron import (
    AbsorbTransposeIntoLUTNeuron,
)
from finn.transformation.fpgadataflow.absorb_lutneuron_output_transpose import (
    AbsorbLUTNeuronOutputTranspose,
)


def test_absorb_transpose_into_lutneuron_remaps_indices():
    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 2, 2, 3])
    transposed = helper.make_tensor_value_info("T", TensorProto.UINT8, [1, 3, 2, 2])
    reshaped = helper.make_tensor_value_info("R", TensorProto.UINT8, [1, 12])
    output = helper.make_tensor_value_info("Y", TensorProto.UINT8, [1, 2])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 12])
    indices = helper.make_tensor("indices", TensorProto.INT64, [2, 2], [0, 1, 5, 11])
    table = helper.make_tensor("table", TensorProto.UINT8, [2, 4], [0, 1, 1, 0, 1, 0, 0, 1])
    transpose = helper.make_node("Transpose", ["X"], ["T"], perm=[0, 3, 1, 2])
    reshape = helper.make_node(
        "Reshape",
        ["T", "shape"],
        ["R"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inp_shape=[1, 3, 2, 2],
        out_shape=[1, 12],
        dtype="UINT8",
        PE=1,
    )
    lut = helper.make_node(
        "LUTNeuron",
        ["R", "indices", "table"],
        ["Y"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumInputs=12,
        NumNeurons=2,
        FanIn=2,
        InputType="UINT8",
        OutputType="BINARY",
        numInputVectors=[1],
    )
    graph = helper.make_graph(
        [transpose, reshape, lut],
        "absorb_transpose_lutneuron",
        [inp],
        [output],
        initializer=[shape, indices, table],
        value_info=[transposed, reshaped],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    model = model.transform(AbsorbTransposeIntoLUTNeuron())

    assert not model.get_nodes_by_op_type("Transpose")
    np.testing.assert_array_equal(model.get_initializer("indices"), [[0, 3], [4, 11]])
    assert model.graph.node[0].input[0] == "X"


def test_absorb_lutneuron_output_transpose_reorders_rows():
    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 4])
    reshaped = helper.make_tensor_value_info("R", TensorProto.UINT8, [1, 2, 2])
    transposed = helper.make_tensor_value_info("T", TensorProto.UINT8, [1, 2, 2])
    output = helper.make_tensor_value_info("Y", TensorProto.UINT8, [1, 2, 2])
    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 2, 2])
    indices = helper.make_tensor("indices", TensorProto.INT64, [4, 1], [0, 1, 2, 3])
    table = helper.make_tensor("table", TensorProto.UINT8, [4, 2], list(range(8)))
    reshape = helper.make_node(
        "Reshape",
        ["L", "shape"],
        ["R"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inp_shape=[1, 4],
        out_shape=[1, 2, 2],
        dtype="UINT8",
        PE=1,
    )
    transpose = helper.make_node("Transpose", ["R"], ["T"], perm=[0, 2, 1])
    lut = helper.make_node(
        "LUTNeuron",
        ["X", "indices", "table"],
        ["L"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumInputs=4,
        NumNeurons=4,
        FanIn=1,
        InputType="UINT8",
        OutputType="UINT8",
        numInputVectors=[1],
    )
    identity = helper.make_node("Identity", ["T"], ["Y"])
    graph = helper.make_graph(
        [lut, reshape, transpose, identity],
        "lutneuron_output_transpose",
        [inp],
        [output],
        initializer=[shape, indices, table],
        value_info=[reshaped, transposed],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    model = model.transform(AbsorbLUTNeuronOutputTranspose())

    assert not model.get_nodes_by_op_type("Transpose")
    np.testing.assert_array_equal(model.get_initializer("indices"), [[0], [2], [1], [3]])
    np.testing.assert_array_equal(model.get_initializer("table"), [[0, 1], [4, 5], [2, 3], [6, 7]])
