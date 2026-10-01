"""Absorb an output transpose into the preceding LUTNeuron row order."""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class AbsorbLUTNeuronOutputTranspose(Transformation):
    """Remove ``LUTNeuron -> Reshape -> Transpose`` by reordering LUT rows."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph_modified = False
        for transpose in list(model.graph.node):
            if transpose.op_type != "Transpose":
                continue
            reshape = model.find_producer(transpose.input[0])
            if reshape is None or not reshape.op_type.startswith("Reshape"):
                continue
            perm_attr = get_by_name(transpose.attribute, "perm")
            if perm_attr is None:
                continue
            perm = list(perm_attr.ints)
            old_shape = model.get_tensor_shape(reshape.output[0])
            new_shape = model.get_tensor_shape(transpose.output[0])
            if old_shape is None or new_shape is None or len(perm) != len(old_shape):
                continue

            source = reshape.input[0]
            while True:
                producer = model.find_producer(source)
                if producer is None or producer.op_type != "Cast":
                    break
                source = producer.input[0]
            lut = model.find_producer(source)
            if lut is None or not lut.op_type.startswith("LUTNeuron"):
                continue
            indices = model.get_initializer(lut.input[1])
            table = model.get_initializer(lut.input[2])
            if indices is None or table is None or indices.ndim != 2 or table.ndim != 2:
                continue
            if np.prod(old_shape) != np.prod(new_shape) or indices.shape[0] != np.prod(old_shape):
                continue

            row_order = []
            for new_flat in range(int(np.prod(new_shape))):
                new_coord = np.unravel_index(new_flat, new_shape)
                old_coord = [0] * len(old_shape)
                for new_axis, old_axis in enumerate(perm):
                    old_coord[old_axis] = new_coord[new_axis]
                row_order.append(np.ravel_multi_index(tuple(old_coord), old_shape))
            model.set_initializer(lut.input[1], indices[np.asarray(row_order)].copy())
            model.set_initializer(lut.input[2], table[np.asarray(row_order)].copy())

            if reshape.domain == "finn.custom_op.fpgadataflow":
                reshape_inst = getCustomOp(reshape)
                reshape_inst.set_nodeattr("out_shape", list(new_shape))
            model.set_tensor_shape(reshape.output[0], list(new_shape))
            consumers = model.find_consumers(transpose.output[0])
            for consumer in consumers:
                for input_index, input_name in enumerate(consumer.input):
                    if input_name == transpose.output[0]:
                        consumer.input[input_index] = reshape.output[0]
            graph_modified = True
            graph = model.graph
            graph.node.remove(transpose)

        return model, graph_modified
