# Copyright (c) 2026 Norwegian University of Science and Technology (NTNU)
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


def _smallest_dtype(values):
    values = np.asarray(values)
    min_value = int(values.min())
    max_value = int(values.max())
    if min_value < 0:
        value = min_value if abs(min_value) > max_value else -max_value - 1
    else:
        value = max_value
    return DataType.get_smallest_possible(value)


def _find_delta_encoding(thresholds):
    """Return base, step and one-bit cumulative-error increments for one row."""
    row = np.asarray(thresholds)
    if row.ndim != 1 or row.size == 0:
        return None
    if not np.issubdtype(row.dtype, np.integer):
        if not np.equal(row, np.rint(row)).all():
            return None
        row = np.rint(row).astype(np.int64)
    else:
        row = row.astype(np.int64)

    base = int(row[0])
    if row.size == 1:
        return base, 0, np.zeros(1, dtype=np.uint8)
    differences = np.diff(row)
    step = int(differences.min())
    errors = differences - step
    if np.isin(errors, (0, 1)).all():
        return base, step, np.concatenate(([0], errors)).astype(np.uint8)
    return None


class DeltaCompressThresholds(Transformation):
    """Promote representable Thresholding_rtl nodes to DeltaThresholding_rtl."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_index = 0
        for node in list(graph.node):
            node_index += 1
            if node.op_type != "Thresholding_rtl":
                continue

            inst = getCustomOp(node)
            if inst.get_nodeattr("runtime_writeable_weights") == 1:
                continue

            thresholds = model.get_initializer(node.input[1])
            if thresholds is None or thresholds.ndim != 2:
                continue

            thresholds, actval, weight_dtype = inst.get_rtl_thresholds(thresholds)

            if thresholds.shape[0] == 1:
                thresholds = np.tile(thresholds, (inst.get_nodeattr("NumChannels"), 1))
            if thresholds.shape[0] != inst.get_nodeattr("NumChannels"):
                continue

            max_input = inst.get_input_datatype().max()
            retained = []
            counts = []
            for row in thresholds:
                row = np.asarray(row)
                count = row.size
                while count > 0 and row[count - 1] > max_input:
                    count -= 1
                if count == 0:
                    retained.append(np.asarray([0], dtype=np.int64))
                else:
                    retained.append(row[:count])
                counts.append(count)

            max_steps = max(1, max(counts))
            encodings = [_find_delta_encoding(row) for row in retained]
            if any(encoding is None for encoding in encodings):
                continue

            bases = np.asarray([encoding[0] for encoding in encodings], dtype=np.int64)
            steps = np.asarray([encoding[1] for encoding in encodings], dtype=np.int64)
            errors = np.zeros((len(encodings), max_steps), dtype=np.uint8)
            for row_index, encoding in enumerate(encodings):
                errors[row_index, : len(encoding[2])] = encoding[2]
            counts = np.asarray(counts, dtype=np.int64)

            node_prefix = node.name if node.name else f"DeltaThresholding_{node_index}"
            base_name = f"{node_prefix}_base"
            step_name = f"{node_prefix}_step"
            error_name = f"{node_prefix}_error"
            count_name = f"{node_prefix}_count"
            model.set_initializer(base_name, bases)
            model.set_initializer(step_name, steps)
            model.set_initializer(error_name, errors)
            model.set_initializer(count_name, counts)
            model.set_tensor_datatype(base_name, _smallest_dtype(bases))
            model.set_tensor_datatype(step_name, _smallest_dtype(steps))
            model.set_tensor_datatype(error_name, DataType["UINT1"])
            model.set_tensor_datatype(count_name, DataType.get_smallest_possible(max_steps))

            new_node = helper.make_node(
                "DeltaThresholding_rtl",
                [node.input[0], base_name, step_name, error_name, count_name],
                list(node.output),
                domain="finn.custom_op.fpgadataflow.rtl",
                name=node_prefix,
            )
            for attribute in node.attribute:
                new_node.attribute.append(attribute)
            new_inst = getCustomOp(new_node)
            new_inst.set_nodeattr("ActVal", int(actval))
            new_inst.set_nodeattr("numSteps", int(max_steps))
            new_inst.set_nodeattr("weightDataType", weight_dtype.name)
            graph.node.insert(node_index, new_node)
            graph.node.remove(node)
            graph_modified = True

        return model, graph_modified