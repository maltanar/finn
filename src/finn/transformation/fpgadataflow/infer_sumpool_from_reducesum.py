"""Lower the CLGN last-axis reduction to a FINN SumPool layer."""

import numpy as np
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class InferSumPoolFromReduceSum(Transformation):
    """Convert a binary LUT tail ending in ``ReduceSum(axis=-1)``."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        graph_modified = False
        for reduce_node in list(graph.node):
            if reduce_node.op_type != "ReduceSum":
                continue
            second_cast = model.find_producer(reduce_node.input[0])
            reshape_input = (
                second_cast.input[0]
                if second_cast is not None and second_cast.op_type == "Cast"
                else reduce_node.input[0]
            )
            reshape = model.find_producer(reshape_input)
            if reshape is None or not reshape.op_type.startswith("Reshape"):
                continue
            if model.get_tensor_shape(reshape.output[0]) != [1, 10, 128]:
                continue
            if model.get_tensor_shape(reduce_node.output[0]) != [1, 10]:
                continue
            first_cast = model.find_producer(reshape.input[0])
            lut_input = (
                first_cast.input[0]
                if first_cast is not None and first_cast.op_type == "Cast"
                else reshape.input[0]
            )
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
            reshape_inst.set_nodeattr("out_shape", [1, 1, 1, 1280])
            model.set_tensor_shape(reshape.output[0], [1, 1, 1, 1280])
            if first_cast is not None and first_cast.op_type == "Cast":
                reshape.input[0] = lut.output[0]
                graph.node.remove(first_cast)
            if second_cast is not None and second_cast.op_type == "Cast":
                graph.node.remove(second_cast)

            pool_name = model.make_new_valueinfo_name()
            pool = helper.make_node(
                "Pool",
                [reshape.output[0]],
                [pool_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                Channels=10,
                PE=10,
                KernelSize=[1, 128],
                Function="SumPool",
                OutImgDims=[1, 1],
                InputDataType=input_dtype.name,
                OutputDataType="UINT8",
                AccumBits=8,
                Size=0,
                BatchSize=1,
                cpp_interface="hls_vector",
                name="SumPool_" + reduce_node.name,
            )
            model.set_tensor_shape(pool_name, [1, 1, 1, 10])
            model.set_tensor_datatype(pool_name, DataType["UINT8"])
            model.set_tensor_datatype(reduce_node.output[0], DataType["UINT8"])

            final_reshape = helper.make_node(
                "Reshape",
                [pool_name],
                [reduce_node.output[0]],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                inp_shape=[1, 1, 1, 10],
                out_shape=[1, 10],
                dtype="UINT8",
                PE=1,
                name="SumPoolOutputReshape_" + reduce_node.name,
            )
            graph_index = list(graph.node).index(reduce_node)
            graph.node.insert(graph_index, pool)
            graph.node.insert(graph_index + 1, final_reshape)
            graph.node.remove(reduce_node)
            graph_modified = True

        return model, graph_modified
