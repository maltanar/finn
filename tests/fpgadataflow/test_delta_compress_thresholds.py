# Copyright (c) 2026 Norwegian University of Science and Technology (NTNU)
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe

from finn.transformation.fpgadataflow.delta_compress_thresholds import (
    DeltaCompressThresholds,
    _find_delta_encoding,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def make_thresholding_model(
    thresholds,
    output_dtype="UINT3",
    input_dtype="INT8",
    threshold_dtype="INT8",
    pe=1,
    actval=0,
):
    channels = thresholds.shape[0]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, channels])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, channels])
    node = helper.make_node(
        "Thresholding_rtl",
        ["inp", "thresh"],
        ["out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=channels,
        PE=pe,
        numSteps=thresholds.shape[1],
        inputDataType=input_dtype,
        weightDataType=threshold_dtype,
        outputDataType=output_dtype,
        numInputVectors=[1],
        ActVal=actval,
        name="thresholding",
    )
    graph = helper.make_graph([node], "delta_thresholding", [inp], [out])
    model = ModelWrapper(helper.make_model(graph, producer_name="delta-thresholding-test"))
    model.set_tensor_datatype("inp", DataType[input_dtype])
    model.set_tensor_datatype("out", DataType[output_dtype])
    model.set_tensor_datatype("thresh", DataType[threshold_dtype])
    model.set_initializer("thresh", thresholds)
    return model


def make_sweep_thresholds(output_dtype, threshold_dtype, input_dtype, narrow):
    output_dt = DataType[output_dtype]
    threshold_dt = DataType[threshold_dtype]
    full_steps = output_dt.get_num_possible_values() - 1
    if output_dt.signed():
        effective = np.arange(full_steps, dtype=np.int64) + threshold_dt.min()
        thresholds = effective[1:] if narrow else effective
    else:
        terminal = threshold_dt.max() + (
            1 if narrow and not DataType[input_dtype].signed() else 0
        )
        effective = terminal - np.arange(full_steps - 1, -1, -1, dtype=np.int64)
        thresholds = effective[:-1] if narrow else effective
    assert np.vectorize(threshold_dt.allowed)(thresholds).all()
    return np.tile(thresholds, (4, 1))


def test_find_delta_encoding_example():
    thresholds = np.array([15, 17, 20, 22, 24, 27, 29], dtype=np.int64)
    base, step, errors = _find_delta_encoding(thresholds)
    assert base == 15
    assert step == 2
    assert errors.tolist() == [0, 0, 1, 0, 0, 1, 0]


def test_delta_compress_promotes_node_and_preserves_values():
    thresholds = np.array(
        [[15, 17, 20, 22, 24, 27, 29], [4, 7, 9, 11, 13, 16, 18]], dtype=np.int64
    )
    model = make_thresholding_model(thresholds)
    model = model.transform(DeltaCompressThresholds())

    node = model.graph.node[0]
    assert node.op_type == "DeltaThresholding_rtl"
    assert len(node.input) == 5
    np.testing.assert_array_equal(model.get_initializer(node.input[1]), [15, 4])
    np.testing.assert_array_equal(model.get_initializer(node.input[2]), [2, 2])
    np.testing.assert_array_equal(
        model.get_initializer(node.input[3]),
        [[0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0]],
    )


def test_delta_compress_node_execution_matches_original():
    thresholds = np.array(
        [
            [15, 17, 20, 22, 24, 27, 29],
            [4, 7, 9, 11, 13, 16, 18],
            [15, 17, 20, 22, 24, 27, 29],
            [4, 7, 9, 11, 13, 16, 18],
        ],
        dtype=np.int64,
    )
    model_before = make_thresholding_model(thresholds, pe=2)
    model_before = model_before.transform(SetExecMode("cppsim"))
    input_values = gen_finn_dt_tensor(DataType["INT8"], (1, 4))
    input_dict = {"inp": input_values}
    expected = oxe.execute_onnx(model_before, input_dict)["out"]

    model_after = model_before.transform(DeltaCompressThresholds())
    actual = oxe.execute_onnx(model_after, input_dict)["out"]

    assert model_after.graph.node[0].op_type == "DeltaThresholding_rtl"
    np.testing.assert_array_equal(actual, expected)


def test_delta_thresholding_rtlsim_matches_python_reference():
    thresholds = np.array(
        [
            [15, 17, 20, 22, 24, 27, 29],
            [4, 7, 9, 11, 13, 16, 18],
            [15, 17, 20, 22, 24, 27, 29],
            [4, 7, 9, 11, 13, 16, 18],
        ],
        dtype=np.int64,
    )
    model_reference = make_thresholding_model(thresholds, pe=2)
    model_reference = model_reference.transform(SetExecMode("cppsim"))
    input_values = gen_finn_dt_tensor(DataType["INT8"], (1, 4))
    input_dict = {"inp": input_values}
    expected = oxe.execute_onnx(model_reference, input_dict)["out"]

    model = model_reference.transform(DeltaCompressThresholds())
    assert model.graph.node[0].op_type == "DeltaThresholding_rtl"
    model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 5))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    actual = oxe.execute_onnx(model, input_dict)["out"]

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "output_dtype, threshold_dtype, input_dtype, narrow",
    [
        ("UINT2", "UINT8", "UINT8", False),
        ("UINT3", "UINT8", "INT8", True),
        ("INT2", "INT8", "INT8", False),
        ("INT3", "INT8", "UINT8", True),
    ],
)
@pytest.mark.parametrize("pe", [1, 2, 4])
def test_delta_compress_configuration_sweep(
    output_dtype, threshold_dtype, input_dtype, narrow, pe
):
    thresholds = make_sweep_thresholds(output_dtype, threshold_dtype, input_dtype, narrow)
    act = DataType[output_dtype]
    actval = act.min()
    if narrow and act.signed():
        actval += 1
    model_before = make_thresholding_model(
        thresholds,
        output_dtype=output_dtype,
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
        pe=pe,
        actval=actval,
    )
    model_before = model_before.transform(SetExecMode("cppsim"))
    input_values = gen_finn_dt_tensor(DataType[input_dtype], (1, 4))
    input_dict = {"inp": input_values}
    expected = oxe.execute_onnx(model_before, input_dict)["out"]

    model_after = model_before.transform(DeltaCompressThresholds())
    actual = oxe.execute_onnx(model_after, input_dict)["out"]

    assert model_after.graph.node[0].op_type == "DeltaThresholding_rtl"
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "output_dtype, threshold_dtype, input_dtype, narrow",
    [
        ("UINT2", "UINT8", "UINT8", False),
        ("UINT3", "UINT8", "INT8", True),
        ("INT2", "INT8", "INT8", False),
        ("INT3", "INT8", "UINT8", True),
    ],
)
@pytest.mark.parametrize("pe", [1, 2, 4])
def test_delta_thresholding_rtlsim_configuration_sweep(
    output_dtype, threshold_dtype, input_dtype, narrow, pe
):
    thresholds = make_sweep_thresholds(output_dtype, threshold_dtype, input_dtype, narrow)
    act = DataType[output_dtype]
    actval = act.min()
    if narrow and act.signed():
        actval += 1
    model_reference = make_thresholding_model(
        thresholds,
        output_dtype=output_dtype,
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
        pe=pe,
        actval=actval,
    )
    model_reference = model_reference.transform(SetExecMode("cppsim"))
    input_values = gen_finn_dt_tensor(DataType[input_dtype], (1, 4))
    input_dict = {"inp": input_values}
    expected = oxe.execute_onnx(model_reference, input_dict)["out"]

    model = model_reference.transform(DeltaCompressThresholds())
    assert model.graph.node[0].op_type == "DeltaThresholding_rtl"
    model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 5))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    actual = oxe.execute_onnx(model, input_dict)["out"]

    np.testing.assert_array_equal(actual, expected)


def test_delta_compress_falls_back_when_residual_is_not_one_bit():
    thresholds = np.array([[15, 17, 20, 23, 24, 25, 27]], dtype=np.int64)
    model = make_thresholding_model(thresholds)
    model = model.transform(DeltaCompressThresholds())
    assert model.graph.node[0].op_type == "Thresholding_rtl"


def test_delta_compress_uses_rtl_narrow_range_normalization():
    thresholds = np.array([[-127, -126]], dtype=np.int64)
    model = make_thresholding_model(thresholds, output_dtype="INT2")
    model = model.transform(DeltaCompressThresholds())

    node = model.graph.node[0]
    assert node.op_type == "DeltaThresholding_rtl"
    assert node.input[3].endswith("_error")
    inst = getCustomOp(node)
    assert inst.get_nodeattr("numSteps") == 3
    assert inst.get_nodeattr("ActVal") == -1


def test_delta_thresholding_generates_static_parameter_files():
    thresholds = np.array([[15, 17, 20, 22, 24, 27, 29]], dtype=np.int64)
    model = make_thresholding_model(thresholds)
    model = model.transform(DeltaCompressThresholds())
    node = model.graph.node[0]
    model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 5))
    code_gen_dir = getCustomOp(model.graph.node[0]).get_nodeattr("code_gen_dir_ipgen")

    assert os.path.exists(os.path.join(code_gen_dir, "thresholding_base_0.dat"))
    assert os.path.exists(os.path.join(code_gen_dir, "thresholding_step_0.dat"))
    assert os.path.exists(os.path.join(code_gen_dir, "thresholding_error_0.dat"))
    wrapper = open(os.path.join(code_gen_dir, "thresholding.v")).read()
    assert "delta_thresholding" in open(
        os.path.join(code_gen_dir, "delta_thresholding.sv")
    ).read()
    assert "$MODULE_NAME_AXI_WRAPPER$" not in wrapper
    assert "$BASE_PATH$" not in wrapper
    assert "BASE_SIGNED = 0" in wrapper
    assert "STEP_SIGNED = 0" in wrapper
    assert "parameter EW = 3" in wrapper


def test_delta_thresholding_supports_signed_thresholds_with_unsigned_input():
    thresholds = np.array([[-5, -3, -1, 1, 3, 5, 7]], dtype=np.int64)
    model = make_thresholding_model(
        thresholds, input_dtype="UINT8", threshold_dtype="INT8"
    )
    model = model.transform(DeltaCompressThresholds())
    node = model.graph.node[0]
    assert node.op_type == "DeltaThresholding_rtl"
    model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 5))
    code_gen_dir = getCustomOp(model.graph.node[0]).get_nodeattr("code_gen_dir_ipgen")

    wrapper = open(os.path.join(code_gen_dir, "thresholding.v")).read()
    assert "SIGNED = 0" in wrapper
    assert "BASE_SIGNED = 1" in wrapper
    assert "STEP_SIGNED = 0" in wrapper


def test_delta_compress_multiple_nodes_no_collision():
    # Two thresholding layers with different channels and step counts
    thresh0 = np.array([[10, 12, 14, 16], [20, 22, 25, 27]], dtype=np.int64)  # 2 channels, 4 steps
    thresh1 = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64)                # 2 channels, 3 steps

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 2])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])

    # Node 0: channels=2, steps=4
    node0 = helper.make_node(
        "Thresholding_rtl",
        ["inp", "thresh0"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=2,
        PE=1,
        numSteps=4,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="UINT3",
        numInputVectors=[1],
        ActVal=0,
        name="",  # empty name as after SpecializeLayers
    )
    # Node 1: channels=2, steps=3
    node1 = helper.make_node(
        "Thresholding_rtl",
        ["mid", "thresh1"],
        ["out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=2,
        PE=1,
        numSteps=3,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="UINT3",
        numInputVectors=[1],
        ActVal=0,
        name="",  # empty name as after SpecializeLayers
    )

    graph = helper.make_graph([node0, node1], "multi_node_test", [inp], [out])
    model = ModelWrapper(helper.make_model(graph, producer_name="multi-test"))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("mid", DataType["UINT3"])
    model.set_tensor_datatype("out", DataType["UINT3"])
    model.set_tensor_datatype("thresh0", DataType["INT8"])
    model.set_initializer("thresh0", thresh0)
    model.set_tensor_datatype("thresh1", DataType["INT8"])
    model.set_initializer("thresh1", thresh1)

    model_comp = model.transform(DeltaCompressThresholds())

    assert len(model_comp.graph.node) == 2
    n0 = model_comp.graph.node[0]
    n1 = model_comp.graph.node[1]
    assert n0.op_type == "DeltaThresholding_rtl"
    assert n1.op_type == "DeltaThresholding_rtl"

    # Verify input parameter tensors are distinct
    assert n0.input[1] != n1.input[1]
    assert n0.input[2] != n1.input[2]
    assert n0.input[3] != n1.input[3]
    assert n0.input[4] != n1.input[4]

    # Verify parameters were not overwritten
    np.testing.assert_array_equal(model_comp.get_initializer(n0.input[1]), [10, 20])
    np.testing.assert_array_equal(model_comp.get_initializer(n1.input[1]), [1, 2])
    assert model_comp.get_initializer(n0.input[3]).shape == (2, 4)
    assert model_comp.get_initializer(n1.input[3]).shape == (2, 3)