"""Lower the CLGN last-axis reduction to FINN GlobalAccPool."""

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class InferGlobalAccPoolFromReduceSum(Transformation):
    """Convert a binary LUT tail ending in ``ReduceSum(axis=-1)``."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        graph_modified = False
        for reduce_node in list(graph.node):
            if reduce_node.op_type != "ReduceSum":
                continue
            second_cast = model.find_producer(reduce_node.input[0])
            reshape_input = second_cast.input[0] if second_cast is not None and second_cast.op_type == "Cast" else reduce_node.input[0]
            reshape = model.find_producer(reshape_input)
            if reshape is None or not reshape.op_type.startswith("Reshape"):
                continue
            old_shape = model.get_tensor_shape(reshape.output[0])
            output_shape = model.get_tensor_shape(reduce_node.output[0])
            if old_shape != [1, 10, 128] or output_shape != [1, 10]:
                continue
            first_cast = model.find_producer(reshape.input[0])
            lut_input = first_cast.input[0] if first_cast is not None and first_cast.op_type == "Cast" else reshape.input[0]
            lut = model.find_producer(lut_input)
            if lut is None or not lut.op_type.startswith("LUTNeuron"):
                continue
            indices = model.get_initializer(lut.input[1])
            table = model.get_initializer(lut.input[2])
            if indices is None or table is None or indices.shape[0] != 1280:
                continue

            row_order = np.asarray([(row % 10) * 128 + row // 10 for row in range(1280)])
            model.set_initializer(lut.input[1], indices[row_order].copy())
            model.set_initializer(lut.input[2], table[row_order].copy())

            input_dtype = model.get_tensor_datatype(lut.output[0])
            reshape_inst = getCustomOp(reshape)
            reshape_inst.set_nodeattr("dtype", input_dtype.name)
            reshape_inst.set_nodeattr("out_shape", [1, 128, 10])
            model.set_tensor_shape(reshape.output[0], [1, 128, 10])
            if first_cast is not None and first_cast.op_type == "Cast":
                reshape.input[0] = lut.output[0]
                graph.node.remove(first_cast)
            if second_cast is not None and second_cast.op_type == "Cast":
                graph.node.remove(second_cast)

            packed_name = model.make_new_valueinfo_name()
            packed_reshape = helper.make_node(
                "Reshape",
                [reshape.output[0]],
                [packed_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                inp_shape=[1, 128, 10],
                out_shape=[1, 1, 128, 10],
                dtype=input_dtype.name,
                PE=1,
                name="GlobalAccPoolInputReshape_" + reduce_node.name,
            )
            model.set_tensor_shape(packed_name, [1, 1, 128, 10])
            model.set_tensor_datatype(packed_name, input_dtype)

            pool_name = model.make_new_valueinfo_name()
            pool = helper.make_node(
                "GlobalAccPool",
                [packed_name],
                [pool_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                NumChannels=10,
                PE=1,
                inputDataType=input_dtype.name,
                numInputVectors=[1, 1, 128],
                name="GlobalAccPool_" + reduce_node.name,
            )
            pool_inst = getCustomOp(pool)
            pool_dtype = pool_inst.get_output_datatype()
            model.set_tensor_shape(pool_name, [1, 1, 1, 10])
            model.set_tensor_datatype(pool_name, pool_dtype)

            final_reshape = helper.make_node(
                "Reshape",
                [pool_name],
                [reduce_node.output[0]],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                inp_shape=[1, 1, 1, 10],
                out_shape=[1, 10],
                dtype=pool_dtype.name,
                PE=1,
                name="GlobalAccPoolOutputReshape_" + reduce_node.name,
            )
            graph_index = list(graph.node).index(reduce_node)
            graph.node.insert(graph_index, packed_reshape)
            graph.node.insert(graph_index + 1, pool)
            graph.node.insert(graph_index + 2, final_reshape)
            graph.node.remove(reduce_node)
            graph_modified = True

        return model, graph_modified
