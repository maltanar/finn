# Copyright (C) 2026, Norwegian University of Science and Technology (NTNU)
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
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def _make_sumpool_model():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1, 1, 1280])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 1, 10])
    node = helper.make_node(
        "Pool",
        ["inp"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Channels=10,
        PE=10,
        KernelSize=[1, 128],
        Function="SumPool",
        OutImgDims=[1, 1],
        InputDataType="BINARY",
        OutputDataType="UINT8",
        AccumBits=8,
        Size=0,
        BatchSize=1,
        cpp_interface="hls_vector",
    )
    graph = helper.make_graph([node], "sumpool", [inp], [out])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="sumpool-test"))
    model.set_tensor_datatype("inp", DataType["BINARY"])
    return model


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
def test_fpgadataflow_sumpool(exec_mode):
    model = _make_sumpool_model()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(1, 1, 1, 1280)).astype(np.float32)
    expected = x.reshape(1, 1, 1, 128, 10).sum(axis=3)

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    else:
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    actual = oxe.execute_onnx(model, {"inp": x})["out"]
    np.testing.assert_array_equal(actual, expected), exec_mode + " failed"