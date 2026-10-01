"""Absorb a transpose before a reshape into LUTNeuron indices."""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class AbsorbTransposeIntoLUTNeuron(Transformation):
    """Remove ``Transpose -> Reshape`` before a LUTNeuron by remapping indices."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        graph_modified = False
        for lut_node in list(graph.node):
            if not lut_node.op_type.startswith("LUTNeuron"):
                continue
            reshape_node = model.find_producer(lut_node.input[0])
            if reshape_node is None or not reshape_node.op_type.startswith("Reshape"):
                continue
            transpose_node = model.find_producer(reshape_node.input[0])
            if transpose_node is None or transpose_node.op_type != "Transpose":
                continue
            if len(model.find_consumers(transpose_node.output[0])) != 1:
                continue

            input_shape = model.get_tensor_shape(transpose_node.input[0])
            transposed_shape = model.get_tensor_shape(transpose_node.output[0])
            reshaped_shape = model.get_tensor_shape(reshape_node.output[0])
            indices = model.get_initializer(lut_node.input[1])
            if input_shape is None or transposed_shape is None or reshaped_shape is None:
                continue
            if indices is None or indices.ndim != 2:
                continue
            if np.prod(transposed_shape) != np.prod(reshaped_shape):
                continue
            if len(input_shape) != len(transposed_shape):
                continue

            perm_attr = get_by_name(transpose_node.attribute, "perm")
            if perm_attr is None or len(perm_attr.ints) != len(input_shape):
                continue
            perm = list(perm_attr.ints)
            if sorted(perm) != list(range(len(input_shape))):
                continue

            flat_indices = indices.astype(np.int64, copy=False)
            if flat_indices.min() < 0 or flat_indices.max() >= np.prod(transposed_shape):
                continue
            remapped = np.empty_like(flat_indices)
            for position, flat_index in np.ndenumerate(flat_indices):
                output_coord = np.unravel_index(int(flat_index), transposed_shape)
                input_coord = [0] * len(input_shape)
                for output_axis, input_axis in enumerate(perm):
                    input_coord[input_axis] = output_coord[output_axis]
                remapped[position] = np.ravel_multi_index(tuple(input_coord), input_shape)

            model.set_initializer(lut_node.input[1], remapped.astype(indices.dtype))
            reshape_node.input[0] = transpose_node.input[0]
            if reshape_node.domain == "finn.custom_op.fpgadataflow":
                getCustomOp(reshape_node).set_nodeattr("inp_shape", list(input_shape))
            graph.node.remove(transpose_node)
            graph_modified = True

        return model, graph_modified
