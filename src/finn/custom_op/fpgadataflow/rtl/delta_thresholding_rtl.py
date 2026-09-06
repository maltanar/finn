# Copyright (c) 2026 Norwegian University of Science and Technology (NTNU)
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil

import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
from finn.transformation.fpgadataflow.delta_compress_thresholds import _smallest_dtype
from finn.util.data_packing import pack_innermost_dim_as_hex_string
from finn.util.settings import get_settings


class DeltaThresholding_rtl(Thresholding_rtl):
    """RTL thresholding using per-channel base, step and one-bit errors."""

    def _get_compressed_parameters(self, model):
        return (
            model.get_initializer(self.onnx_node.input[1]),
            model.get_initializer(self.onnx_node.input[2]),
            model.get_initializer(self.onnx_node.input[3]),
            model.get_initializer(self.onnx_node.input[4]),
        )

    def minimize_accumulator_width(self, model):
        """Compressed parameters already retain the source threshold dtype."""
        return self.get_weight_datatype()

    def _write_parameter_files(self, model, code_gen_dir):
        bases, steps, errors, counts = self._get_compressed_parameters(model)
        pe = self.get_nodeattr("PE")
        channels = self.get_nodeattr("NumChannels")
        channel_fold = channels // pe
        base_dtype = _smallest_dtype(bases)
        step_dtype = _smallest_dtype(steps)

        def write_values(filename, values, dtype):
            width = max(1, ((dtype.bitwidth() + 3) // 4) * 4)
            packed = pack_innermost_dim_as_hex_string(
                np.asarray(values).reshape(-1, 1), dtype, width, prefix=""
            )
            with open(filename, "w") as parameter_file:
                for value in packed.reshape(-1):
                    parameter_file.write(str(value) + "\n")

        for pe_value in range(pe):
            channel_indices = [fold * pe + pe_value for fold in range(channel_fold)]
            write_values(
                os.path.join(code_gen_dir, f"{self.onnx_node.name}_base_{pe_value}.dat"),
                bases[channel_indices],
                base_dtype,
            )
            write_values(
                os.path.join(code_gen_dir, f"{self.onnx_node.name}_step_{pe_value}.dat"),
                steps[channel_indices],
                step_dtype,
            )
            write_values(
                os.path.join(code_gen_dir, f"{self.onnx_node.name}_error_{pe_value}.dat"),
                errors[channel_indices].reshape(-1),
                DataType["UINT1"],
            )
            write_values(
                os.path.join(code_gen_dir, f"{self.onnx_node.name}_count_{pe_value}.dat"),
                counts[channel_indices],
                _smallest_dtype(counts),
            )
        return base_dtype.bitwidth(), step_dtype.bitwidth()

    def prepare_codegen_rtl_values(self, model):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        base_width, step_width = self._write_parameter_files(model, code_gen_dir)
        bases, steps, _, counts = self._get_compressed_parameters(model)
        base_dtype = _smallest_dtype(bases)
        step_dtype = _smallest_dtype(steps)
        input_dtype = self.get_input_datatype()
        output_dtype = self.get_output_datatype()
        output_width = output_dtype.bitwidth()
        bias = self.get_nodeattr("ActVal")

        if bias >= 0:
            output_bits = int(np.ceil(np.log2(2**output_width + bias)))
        else:
            output_bits = int(
                1
                + np.ceil(
                    np.log2(
                        -bias if -bias >= 2 ** (output_width - 1) else 2**output_width + bias
                    )
                )
            )

        return {
            "$MODULE_NAME_AXI_WRAPPER$": [self.get_verilog_top_module_name()],
            "$TOP_MODULE$": [self.get_verilog_top_module_name()],
            "$N$": [str(output_width)],
            "$O_BITS$": [str(output_bits)],
            "$WI$": [str(input_dtype.bitwidth())],
            "$WT$": [str(self.get_input_datatype(1).bitwidth())],
            "$BW$": [str(base_width)],
            "$SW$": [str(step_width)],
            "$EW$": [str(max(1, int(np.ceil(np.log2(self.get_nodeattr("numSteps") + 1)))))],
            "$C$": [str(self.get_nodeattr("NumChannels"))],
            "$PE$": [str(self.get_nodeattr("PE"))],
            "$NUM_STEPS$": [str(self.get_nodeattr("numSteps"))],
            "$SIGNED$": [str(int(input_dtype.signed()))],
            "$BASE_SIGNED$": [str(int(base_dtype.signed()))],
            "$STEP_SIGNED$": [str(int(step_dtype.signed()))],
            "$BIAS$": [str(bias)],
            "$BASE_PATH$": ['"./%s_base_"' % self.onnx_node.name],
            "$STEP_PATH$": ['"./%s_step_"' % self.onnx_node.name],
            "$ERROR_PATH$": ['"./%s_error_"' % self.onnx_node.name],
            "$COUNT_PATH$": ['"./%s_count_"' % self.onnx_node.name],
            "$CW$": [str(_smallest_dtype(counts).bitwidth())],
        }

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(get_settings().finn_rtllib, "thresholding/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""
        return [
            rtllib_dir + "delta_thresholding.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]

    def get_all_meminit_filenames(self, abspath=False):
        path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        return [
            os.path.join(path, f"{self.onnx_node.name}_{kind}_{pe_value}.dat")
            for kind in ("base", "step", "error", "count")
            for pe_value in range(pe)
        ]

    def generate_hdl(self, model, fpgapart, clk):
        code_gen_dict = self.prepare_codegen_rtl_values(model)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])
        rtlsrc = os.path.join(get_settings().finn_rtllib, "thresholding/hdl")
        with open(os.path.join(rtlsrc, "delta_thresholding_template_wrapper.v")) as template_file:
            wrapper = template_file.read()
        for key, value in code_gen_dict.items():
            wrapper = wrapper.replace(key, "\n".join(value))
        with open(os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"), "w") as wrapper_file:
            wrapper_file.write(wrapper)
        shutil.copy(os.path.join(rtlsrc, "delta_thresholding.sv"), code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def execute_node(self, context, graph):
        bases, steps, errors, counts = (
            context[self.onnx_node.input[1]],
            context[self.onnx_node.input[2]],
            context[self.onnx_node.input[3]],
            context[self.onnx_node.input[4]],
        )
        thresholds = bases[:, None] + np.arange(self.get_nodeattr("numSteps")) * steps[:, None]
        thresholds = thresholds + np.cumsum(errors, axis=1)
        threshold_indices = np.arange(self.get_nodeattr("numSteps"))[None, :]
        thresholds = np.where(threshold_indices < counts[:, None], thresholds, np.inf)
        input_values = context[self.onnx_node.input[0]]
        output = self._execute_thresholds(input_values, thresholds)
        context[self.onnx_node.output[0]] = output.astype(np.float32)

    def _execute_thresholds(self, input_values, thresholds):
        from qonnx.custom_op.general.multithreshold import multithreshold

        is_4d = len(input_values.shape) == 4
        if is_4d:
            input_values = np.transpose(input_values, (0, 3, 1, 2))
        output = multithreshold(
            input_values,
            thresholds,
            out_bias=self.get_nodeattr("ActVal"),
        )
        if is_4d:
            output = output.transpose(0, 2, 3, 1)
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            output = 2 * output - 1
        return output

    def get_op_and_param_counts(self):
        return {
                "param_delta_base_%db" % self.get_input_datatype(1).bitwidth(): self.get_nodeattr(
                    "NumChannels"
                ),
                "param_delta_step_%db" % self.get_input_datatype(1).bitwidth(): self.get_nodeattr(
                    "NumChannels"
                ),
                "param_delta_error_1b": self.get_nodeattr("NumChannels")
                * self.get_nodeattr("numSteps"),
        }

    def get_exp_cycles(self):
        return int(super().get_exp_cycles() * self.get_nodeattr("numSteps"))