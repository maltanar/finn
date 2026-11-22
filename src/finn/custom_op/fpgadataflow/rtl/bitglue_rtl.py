# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import os

from finn.custom_op.fpgadataflow.bitglue import BitGlue
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class BitGlue_rtl(BitGlue, RTLBackend):
    """Zero-LUT combinational AXI-Stream RTL implementation of BitGlue."""

    def get_nodeattr_types(self):
        attrs = BitGlue.get_nodeattr_types(self)
        attrs.update(RTLBackend.get_nodeattr_types(self))
        return attrs

    def execute_node(self, context, graph):
        if self.get_nodeattr("exec_mode") != "rtlsim":
            BitGlue.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model, fpgapart, clk):
        top = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", top)

        n_in = self.get_num_inputs()
        in0_w = self.get_instream_width_padded(0)
        out_w = self.get_outstream_width_padded(0)
        out_bits = self.get_outstream_width(0)

        ports = [
            "    input  wire ap_clk,",
            "    input  wire ap_rst_n,",
            f"    input  wire [{in0_w - 1}:0] in0_V_TDATA,",
            "    input  wire in0_V_TVALID,",
            "    output wire in0_V_TREADY,",
        ]
        if n_in == 2:
            in1_w = self.get_instream_width_padded(1)
            ports.extend([
                f"    input  wire [{in1_w - 1}:0] in1_V_TDATA,",
                "    input  wire in1_V_TVALID,",
                "    output wire in1_V_TREADY,",
            ])
        ports.extend([
            f"    output wire [{out_w - 1}:0] out0_V_TDATA,",
            "    output wire out0_V_TVALID,",
            "    input  wire out0_V_TREADY",
        ])

        lines = [f"module {top} ("] + ports + [");"]
        if n_in == 1:
            lines.append("assign out0_V_TVALID = in0_V_TVALID;")
            lines.append("assign in0_V_TREADY = out0_V_TREADY;")
        else:
            lines.append("assign out0_V_TVALID = in0_V_TVALID & in1_V_TVALID;")
            lines.append("assign in0_V_TREADY = out0_V_TVALID & out0_V_TREADY;")
            lines.append("assign in1_V_TREADY = out0_V_TVALID & out0_V_TREADY;")

        src_stream = self.get_nodeattr("SrcStream")
        src_bit = self.get_nodeattr("SrcBit")

        # Group contiguous runs for clean, compact Verilog
        i = 0
        while i < out_bits:
            s = src_stream[i]
            b = src_bit[i]
            j = i + 1
            if s >= 0:
                while j < out_bits and src_stream[j] == s and src_bit[j] == b + (j - i):
                    j += 1
                run = j - i
                lines.append(
                    f"assign out0_V_TDATA[{i} +: {run}] = in{s}_V_TDATA[{b} +: {run}];"
                )
            else:
                while j < out_bits and src_stream[j] == s:
                    j += 1
                run = j - i
                val_bit = "1'b0" if s == -1 else "1'b1"
                lines.append(
                    f"assign out0_V_TDATA[{i} +: {run}] = {{{run}{{{val_bit}}}}};"
                )
            i = j

        if out_w > out_bits:
            lines.append(f"assign out0_V_TDATA[{out_w - 1}:{out_bits}] = {out_w - out_bits}'d0;")
        lines.append("endmodule")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, f"{top}.v"), "w") as f:
            f.write("\n".join(lines) + "\n")
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""
        top = self.get_nodeattr("gen_top_module")
        return [os.path.join(code_gen_dir, f"{top}.v")]

    def code_generation_ipi(self):
        return [
            f"add_files -norecurse {source}" for source in self.get_rtl_file_list(abspath=True)
        ] + [
            f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        ]
