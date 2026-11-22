# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import os

from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class DuplicateStreams_rtl(DuplicateStreams, RTLBackend):
    """AXI-Stream RTL fork implementation of DuplicateStreams (supports arbitrary bitwidths)."""

    def get_nodeattr_types(self):
        attrs = DuplicateStreams.get_nodeattr_types(self)
        attrs.update(RTLBackend.get_nodeattr_types(self))
        return attrs

    def execute_node(self, context, graph):
        if self.get_nodeattr("exec_mode") != "rtlsim":
            DuplicateStreams.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model, fpgapart, clk):
        top = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", top)

        n_out = self.get_nodeattr("NumOutputStreams")
        w = self.get_instream_width_padded(0)

        ports = [
            "    input  wire ap_clk,",
            "    input  wire ap_rst_n,",
            f"    input  wire [{w - 1}:0] in0_V_TDATA,",
            "    input  wire in0_V_TVALID,",
            "    output wire in0_V_TREADY,",
        ]
        for i in range(n_out):
            ports.extend([
                f"    output wire [{w - 1}:0] out{i}_V_TDATA,",
                f"    output wire out{i}_V_TVALID,",
                f"    input  wire out{i}_V_TREADY" + ("," if i < n_out - 1 else ""),
            ])

        lines = [f"module {top} ("] + ports + [");"]
        lines.append(f"reg [{n_out - 1}:0] done_r;")
        lines.append(f"wire [{n_out - 1}:0] out_rdy;")
        lines.append(f"wire [{n_out - 1}:0] out_vld;")
        for i in range(n_out):
            lines.append(f"assign out{i}_V_TDATA  = in0_V_TDATA;")
            lines.append(f"assign out_vld[{i}]    = in0_V_TVALID & ~done_r[{i}];")
            lines.append(f"assign out{i}_V_TVALID = out_vld[{i}];")
            lines.append(f"assign out_rdy[{i}]    = out{i}_V_TREADY;")
        lines.append("assign in0_V_TREADY = &(done_r | out_rdy);")
        lines.append("always @(posedge ap_clk) begin")
        lines.append("    if (!ap_rst_n) begin")
        lines.append(f"        done_r <= {n_out}'d0;")
        lines.append("    end else if (in0_V_TVALID && in0_V_TREADY) begin")
        lines.append(f"        done_r <= {n_out}'d0;")
        lines.append("    end else begin")
        lines.append("        done_r <= done_r | (out_vld & out_rdy);")
        lines.append("    end")
        lines.append("end")
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
