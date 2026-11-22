# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.

import os

from finn.custom_op.fpgadataflow.abs import Abs
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class Abs_rtl(Abs, RTLBackend):
    """Combinational AXI-Stream RTL implementation of integer Abs."""

    def get_nodeattr_types(self):
        attrs = Abs.get_nodeattr_types(self)
        attrs.update(RTLBackend.get_nodeattr_types(self))
        return attrs

    def execute_node(self, context, graph):
        if self.get_nodeattr("exec_mode") != "rtlsim":
            Abs.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model, fpgapart, clk):
        top = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", top)

        pe = self.get_nodeattr("PE")
        ibw = self.get_input_datatype().bitwidth()
        obw = self.get_output_datatype().bitwidth()
        in_w = self.get_instream_width_padded(0)
        out_w = self.get_outstream_width_padded(0)

        lines = [
            f"module {top} (",
            "    input  wire ap_clk,",
            "    input  wire ap_rst_n,",
            f"    input  wire [{in_w - 1}:0] in0_V_TDATA,",
            "    input  wire in0_V_TVALID,",
            "    output wire in0_V_TREADY,",
            f"    output wire [{out_w - 1}:0] out0_V_TDATA,",
            "    output wire out0_V_TVALID,",
            "    input  wire out0_V_TREADY",
            ");",
            "assign out0_V_TVALID = in0_V_TVALID;",
            "assign in0_V_TREADY = out0_V_TREADY;",
        ]
        for p in range(pe):
            lines.append(f"wire signed [{ibw - 1}:0] lane_{p} = in0_V_TDATA[{p * ibw} +: {ibw}];")
            lines.append(
                f"wire [{ibw - 1}:0] abs_{p} = lane_{p}[{ibw - 1}] ? (-lane_{p}) : lane_{p};"
            )
            if obw >= ibw:
                if obw > ibw:
                    lines.append(
                        f"assign out0_V_TDATA[{p * obw} +: {obw}] = {{{{{obw - ibw}{{1'b0}}}}, abs_{p}}};"
                    )
                else:
                    lines.append(f"assign out0_V_TDATA[{p * obw} +: {obw}] = abs_{p};")
            else:
                lines.append(f"assign out0_V_TDATA[{p * obw} +: {obw}] = abs_{p}[{obw - 1}:0];")

        used_out = pe * obw
        if out_w > used_out:
            lines.append(f"assign out0_V_TDATA[{out_w - 1}:{used_out}] = {out_w - used_out}'d0;")
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
