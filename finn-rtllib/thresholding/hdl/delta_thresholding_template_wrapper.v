// Copyright (c) 2026 Norwegian University of Science and Technology (NTNU)
// SPDX-License-Identifier: BSD-3-Clause
module $MODULE_NAME_AXI_WRAPPER$ #(
    parameter N = $N$,
    parameter O_BITS = $O_BITS$,
    parameter WI = $WI$,
    parameter WT = $WT$,
    parameter BW = $BW$,
    parameter SW = $SW$,
    parameter EW = $EW$,
    parameter CW = $CW$,
    parameter C = $C$,
    parameter PE = $PE$,
    parameter NUM_STEPS = $NUM_STEPS$,
    parameter SIGNED = $SIGNED$,
    parameter BASE_SIGNED = $BASE_SIGNED$,
    parameter STEP_SIGNED = $STEP_SIGNED$,
    parameter BIAS = $BIAS$,
    parameter BASE_PATH = $BASE_PATH$,
    parameter STEP_PATH = $STEP_PATH$,
    parameter ERROR_PATH = $ERROR_PATH$
    , parameter COUNT_PATH = $COUNT_PATH$
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out_V, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input ap_rst_n,
    output in0_V_TREADY,
    input in0_V_TVALID,
    input [((PE*WI+7)/8)*8-1:0] in0_V_TDATA,
    input out_V_TREADY,
    output out_V_TVALID,
    output [((PE*O_BITS+7)/8)*8-1:0] out_V_TDATA
);
    delta_thresholding #(
        .N(N), .O_BITS(O_BITS), .WI(WI), .WT(WT), .BW(BW), .SW(SW),
        .C(C), .PE(PE), .NUM_STEPS(NUM_STEPS), .SIGNED(SIGNED), .EW(EW), .CW(CW),
        .BASE_SIGNED(BASE_SIGNED), .STEP_SIGNED(STEP_SIGNED), .BIAS(BIAS),
        .BASE_PATH(BASE_PATH), .STEP_PATH(STEP_PATH), .ERROR_PATH(ERROR_PATH), .COUNT_PATH(COUNT_PATH)
    ) core (
        .clk(ap_clk), .rst(!ap_rst_n),
        .irdy(in0_V_TREADY), .ivld(in0_V_TVALID), .idat(in0_V_TDATA[PE*WI-1:0]),
        .ordy(out_V_TREADY), .ovld(out_V_TVALID), .odat(out_V_TDATA[PE*O_BITS-1:0])
    );
    generate
        if (((PE*O_BITS+7)/8)*8 > PE*O_BITS)
            assign out_V_TDATA[((PE*O_BITS+7)/8)*8-1:PE*O_BITS] =
                {((((PE*O_BITS+7)/8)*8)-PE*O_BITS){1'b0}};
    endgenerate
endmodule