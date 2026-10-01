"""RTL backend for the hardware Reshape operator."""

import os
import shutil

from finn.custom_op.fpgadataflow.reshape import Reshape
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class Reshape_rtl(Reshape, RTLBackend):
    """Reshape implemented with FINN's AXI data-width converter."""

    def get_nodeattr_types(self):
        attrs = Reshape.get_nodeattr_types(self)
        attrs.update(RTLBackend.get_nodeattr_types(self))
        return attrs

    def execute_node(self, context, graph):
        if self.get_nodeattr("exec_mode") != "rtlsim":
            Reshape.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "dwc", "hdl")
        with open(os.path.join(rtlsrc, "dwc_template.v")) as template_file:
            template = template_file.read()
        top = self.get_verilog_top_module_name()
        for placeholder, value in {
            "TOP_MODULE_NAME": top,
            "IBITS": self.get_instream_width(),
            "OBITS": self.get_outstream_width(),
        }.items():
            template = template.replace(f"${placeholder}$", str(value))
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("gen_top_module", top)
        with open(os.path.join(code_gen_dir, f"{top}.v"), "w") as output_file:
            output_file.write(template)
        shutil.copy(os.path.join(rtlsrc, "dwc.sv"), code_gen_dir)
        shutil.copy(os.path.join(rtlsrc, "dwc_axi.sv"), code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""
        top = self.get_nodeattr("gen_top_module")
        if not isinstance(code_gen_dir, str) or not isinstance(top, str) or not top:
            raise RuntimeError("Reshape RTL code generation attributes are missing")
        return [
            os.path.join(code_gen_dir, "dwc.sv"),
            os.path.join(code_gen_dir, "dwc_axi.sv"),
            os.path.join(code_gen_dir, f"{top}.v"),
        ]

    def code_generation_ipi(self):
        return [
            f"add_files -norecurse {source}" for source in self.get_rtl_file_list(abspath=True)
        ] + [
            "create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        ]
