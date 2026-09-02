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

import numpy as np
import os
import shutil

from finn.custom_op.fpgadataflow.lutneuron import LUTNeuron
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class LUTNeuron_rtl(LUTNeuron, RTLBackend):
    """Fully parallel RTL implementation of LUTNeuron. The whole input frame is
    consumed in a single AXI-Stream beat and all M neurons are evaluated
    combinationally, so `indices` degenerates into wiring and `table` into LUT
    initialization values."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # insert an AXI-Stream register stage in front of the LUT logic
            "input_reg": ("i", False, 0, {0, 1}),
            # insert an AXI-Stream register stage behind the LUT logic
            "output_reg": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(LUTNeuron.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_exp_cycles(self):
        # one beat per input vector, plus the latency of the register stages
        n_vecs = int(np.prod(self.get_nodeattr("numInputVectors")))
        return n_vecs + self.get_nodeattr("input_reg") + self.get_nodeattr("output_reg")

    def _get_indices_table(self, model):
        node = self.onnx_node
        indices = model.get_initializer(node.input[1])
        table = model.get_initializer(node.input[2])
        assert indices is not None, "LUTNeuron requires a constant indices input"
        assert table is not None, "LUTNeuron requires a constant table input"
        m = self.get_nodeattr("NumNeurons")
        k = self.get_nodeattr("FanIn")
        s = self.get_table_size()
        indices = np.asarray(indices).astype(np.int64).reshape(m, k)
        table = np.asarray(table).reshape(m, s)
        c_in = self.get_nodeattr("NumInputs")
        assert indices.min() >= 0 and indices.max() < c_in, "indices out of range [0, C_in)"
        return indices, table

    def _table_literal(self, row, out_bits):
        """Packs one neuron's table row into a single Verilog sized hex literal,
        entry s occupying bits [s*out_bits, (s+1)*out_bits)."""
        mask = (1 << out_bits) - 1
        acc = 0
        for s, val in enumerate(row):
            acc |= (int(val) & mask) << (s * out_bits)
        nbits = len(row) * out_bits
        return "%d'h%0*x" % (nbits, (nbits + 3) // 4, acc)

    def generate_hdl(self, model, fpgapart, clk):
        indices, table = self._get_indices_table(model)
        m = self.get_nodeattr("NumNeurons")
        k = self.get_nodeattr("FanIn")
        ib = self.get_input_bits()
        ob = self.get_out_bits()
        in_bits = self.get_instream_width(0)
        out_bits = self.get_outstream_width(0)
        in_w = self.get_instream_width_padded(0)
        out_w = self.get_outstream_width_padded(0)
        addr_w = k * ib
        input_reg = self.get_nodeattr("input_reg")
        output_reg = self.get_nodeattr("output_reg")
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        body = []
        for neuron in range(m):
            # slot 0 is the least significant part of the address
            slots = [
                "core_in[%d +: %d]" % (int(indices[neuron, slot]) * ib, ib)
                for slot in reversed(range(k))
            ]
            body.append(
                "localparam [%d:0] TBL_%d = %s;"
                % (self.get_table_size() * ob - 1, neuron, self._table_literal(table[neuron], ob))
            )
            body.append(
                "wire [%d:0] ADDR_%d = {%s};" % (addr_w - 1, neuron, ", ".join(slots))
            )
            body.append(
                "assign core_out_raw[%d +: %d] = TBL_%d[ADDR_%d*%d +: %d];"
                % (neuron * ob, ob, neuron, neuron, ob, ob)
            )

        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "IN_STREAM_BITS": in_w,
            "OUT_STREAM_BITS": out_w,
            "IN_BITS": in_bits,
            "OUT_BITS": out_bits,
            "LUT_LOGIC": "\n".join(body),
            "INPUT_STAGE": self._reg_stage(
                "reg_in",
                in_w,
                input_reg,
                ("in0_V_TDATA", "in0_V_TVALID", "in0_V_TREADY"),
                ("core_in_wide", "core_vld", "core_rdy"),
            ),
            "OUTPUT_STAGE": self._reg_stage(
                "reg_out",
                out_w,
                output_reg,
                ("core_out", "core_vld", "core_rdy"),
                ("out0_V_TDATA", "out0_V_TVALID", "out0_V_TREADY"),
            ),
        }

        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/lutneuron"
        with open(rtlsrc + "/lutneuron_template.v", "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            template = template.replace("$%s$" % key_name, str(code_gen_dict[key_name]))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        shutil.copy(rtlsrc + "/lutneuron_skid.sv", code_gen_dir)
        # set ipgen_path and ip_path so that HLSSynthIP and StitchIP do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _reg_stage(self, name, width, enabled, src, dst):
        """Emits either a skid buffer or a straight-through connection between
        the (data, valid, ready) triples `src` and `dst`."""
        sdat, svld, srdy = src
        ddat, dvld, drdy = dst
        if enabled:
            return (
                "lutneuron_skid #(.DATA_WIDTH({w})) {name} (\n"
                "  .clk(ap_clk), .rst(!ap_rst_n),\n"
                "  .idat({sdat}), .ivld({svld}), .irdy({srdy}),\n"
                "  .odat({ddat}), .ovld({dvld}), .ordy({drdy})\n"
                ");"
            ).format(
                w=width,
                name=name,
                sdat=sdat,
                svld=svld,
                srdy=srdy,
                ddat=ddat,
                dvld=dvld,
                drdy=drdy,
            )
        return "assign {ddat} = {sdat};\nassign {dvld} = {svld};\nassign {srdy} = {drdy};".format(
            ddat=ddat, sdat=sdat, dvld=dvld, svld=svld, srdy=srdy, drdy=drdy
        )

    def get_rtl_file_list(self, abspath=False):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/" if abspath else ""
        return [
            code_gen_dir + "lutneuron_skid.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefiles = [
            os.path.join(code_gen_dir, "lutneuron_skid.sv"),
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"),
        ]
        cmd = ["add_files -norecurse %s" % f for f in sourcefiles]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LUTNeuron.execute_node(self, context, graph)
            return
        if mode != "rtlsim":
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        # only input 0 is a stream; indices/table are baked into the HDL
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        exp_ishape = tuple(self.get_normal_input_shape(0))
        exp_oshape = tuple(self.get_normal_output_shape(0))
        inp = context[node.input[0]]
        assert inp.shape == exp_ishape, "Input shape doesn't match expected shape."
        inp_path = os.path.join(code_gen_dir, "input_0.npy")
        np.save(inp_path, inp.reshape(self.get_folded_input_shape(0)))
        rtlsim_inp = npy_to_rtlsim_input(
            inp_path, self.get_input_datatype(0), self.get_instream_width(0)
        )

        io_dict = {"inputs": {"in0": rtlsim_inp}, "outputs": {"out0": []}}
        sim = self.get_rtlsim()
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict)
        self.close_rtlsim(sim)

        odt = self.get_output_datatype(0)
        out_npy_path = os.path.join(code_gen_dir, "output.npy")
        rtlsim_output_to_npy(
            io_dict["outputs"]["out0"],
            out_npy_path,
            odt,
            self.get_folded_output_shape(0),
            self.get_outstream_width(0),
            odt.bitwidth(),
        )
        context[node.output[0]] = np.load(out_npy_path).reshape(exp_oshape).astype(np.float32)
