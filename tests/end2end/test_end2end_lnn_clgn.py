# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

"""Incremental FINN flow tests for the exported CLGN model."""

from pathlib import Path
import os

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_lookuptableconv_to_lookuptable import (
    LowerLookupTableConvToLookupTable,
)
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferConvInpGen,
    InferLUTNeuronLayer,
    InferPool,
    InferReshape,
)
from finn.transformation.fpgadataflow.absorb_transpose_into_lutneuron import (
    AbsorbTransposeIntoLUTNeuron,
)
from finn.transformation.fpgadataflow.infer_globalaccpool_from_reducesum import (
    InferGlobalAccPoolFromReduceSum,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "qonnx-lonnx"
    / "lnn-example"
    / "clgn_tiny_maxpool.onnx"
)


def _checkpoint_dir(stage):
    output_dir = os.environ.get("LNN_CLGN_OUTPUT_DIR")
    if output_dir is None:
        output_dir = os.environ.get("FINN_BUILD_DIR", "/tmp")
    checkpoint_dir = Path(output_dir) / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _write_graph_report(model, report_path):
    with report_path.open("w", encoding="ascii") as report:
        for index, node in enumerate(model.graph.node):
            report.write(
                f"{index}: {node.domain or 'onnx'}::{node.op_type} "
                f"{node.name}\n"
                f"    inputs: {list(node.input)}\n"
                f"    outputs: {list(node.output)}\n"
            )


def _lower_lookup_convs():
    model = ModelWrapper(str(MODEL_PATH))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(LowerLookupTableConvToLookupTable())
    return model.transform(InferShapes()).transform(InferDataTypes())


def _normalize_binary_lookup_datatypes(model):
    for node in model.graph.node:
        if node.op_type != "LookupTable" or node.domain != "qonnx.custom_op.lnn":
            continue
        node_inst = getCustomOp(node)
        if node_inst.get_nodeattr("input_bits") != 1:
            continue
        table = model.get_initializer(node.input[2])
        if table is None or not np.isin(table, [0, 1]).all():
            continue
        model.set_tensor_datatype(node.input[0], DataType["BINARY"])
        model.set_tensor_datatype(node.output[0], DataType["BINARY"])
    return model


def _normalize_binary_im2col_inputs(model):
    for node in model.graph.node:
        if node.op_type == "Im2Col":
            model.set_tensor_datatype(node.input[0], DataType["BINARY"])
    return model


def test_lnn_clgn_lower_lookup_conv():
    model = _lower_lookup_convs()

    assert not any(node.op_type == "LookupTableConv" for node in model.graph.node)
    assert len(model.get_nodes_by_op_type("Im2Col")) == 3
    assert len(model.get_nodes_by_op_type("LookupTable")) == 12

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step1_lower_lookup_conv")
    model.save(str(checkpoint_dir / "lower_lookup_conv.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_convert_lookup_to_lutneuron():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferShapes()).transform(InferDataTypes())

    assert not any(node.op_type == "LookupTable" for node in model.graph.node)
    assert len(model.get_nodes_by_op_type("LUTNeuron")) == 12

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step2_lutneuron")
    model.save(str(checkpoint_dir / "lutneuron.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_convert_streaming_layers():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferPool())
    model = _normalize_binary_im2col_inputs(model)
    model = model.transform(InferConvInpGen())
    model = model.transform(InferShapes()).transform(InferDataTypes())

    assert len(model.get_nodes_by_op_type("LUTNeuron")) == 12
    assert len(model.get_nodes_by_op_type("ConvolutionInputGenerator")) == 6
    assert len(model.get_nodes_by_op_type("Pool")) == 3
    assert not any(node.op_type in ("Im2Col", "MaxPool") for node in model.graph.node)

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step3_streaming_layers")
    model.save(str(checkpoint_dir / "streaming_layers.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_absorb_consecutive_transposes():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferPool())
    model = _normalize_binary_im2col_inputs(model)
    model = model.transform(InferConvInpGen())
    before = len(model.get_nodes_by_op_type("Transpose"))
    model = model.transform(AbsorbConsecutiveTransposes())

    assert len(model.get_nodes_by_op_type("Transpose")) < before
    assert len(model.get_nodes_by_op_type("LUTNeuron")) == 12

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step4_absorb_transposes")
    model.save(str(checkpoint_dir / "absorbed_transposes.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_convert_reshape():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferPool())
    model = _normalize_binary_im2col_inputs(model)
    model = model.transform(InferConvInpGen())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferReshape())

    reshape_nodes = model.get_nodes_by_op_type("Reshape")
    assert len(reshape_nodes) == 2
    assert all(node.domain == "finn.custom_op.fpgadataflow" for node in reshape_nodes)

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    assert len(model.get_nodes_by_op_type("Reshape_rtl")) == 2

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step5_reshape")
    model.save(str(checkpoint_dir / "reshape_rtl.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_absorb_transpose_into_lutneuron():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferPool())
    model = _normalize_binary_im2col_inputs(model)
    model = model.transform(InferConvInpGen())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferReshape())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    before = len(model.get_nodes_by_op_type("Transpose"))
    model = model.transform(AbsorbTransposeIntoLUTNeuron())

    assert len(model.get_nodes_by_op_type("Transpose")) == before - 1
    assert len(model.get_nodes_by_op_type("LUTNeuron_rtl")) == 12
    assert len(model.get_nodes_by_op_type("Reshape_rtl")) == 2

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step6_absorb_transpose_lutneuron")
    model.save(str(checkpoint_dir / "absorbed_transpose_lutneuron.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")


def test_lnn_clgn_lower_reduce_sum_to_global_acc_pool():
    model = _lower_lookup_convs()
    model = _normalize_binary_lookup_datatypes(model)
    model = model.transform(InferLUTNeuronLayer())
    model = model.transform(InferPool())
    model = _normalize_binary_im2col_inputs(model)
    model = model.transform(InferConvInpGen())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferReshape())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    model = model.transform(AbsorbTransposeIntoLUTNeuron())
    model = model.transform(InferGlobalAccPoolFromReduceSum())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    assert len(model.get_nodes_by_op_type("ReduceSum")) == 0
    assert len(model.get_nodes_by_op_type("GlobalAccPool_hls")) == 1
    assert len(model.get_nodes_by_op_type("Reshape_rtl")) == 4

    checkpoint_dir = _checkpoint_dir("lnn_clgn_step7_global_acc_pool")
    model.save(str(checkpoint_dir / "global_acc_pool.onnx"))
    _write_graph_report(model, checkpoint_dir / "graph.txt")
