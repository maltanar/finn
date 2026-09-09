// Copyright (c) 2026 Norwegian University of Science and Technology (NTNU)
// SPDX-License-Identifier: BSD-3-Clause

module delta_thresholding #(
    parameter int unsigned N = 1,
    parameter int unsigned O_BITS = 1,
    parameter int unsigned WI = 1,
    parameter int unsigned WT = 1,
    parameter int unsigned BW = 1,
    parameter int unsigned SW = 1,
    parameter int unsigned EW = 1,
    parameter int unsigned CW = 1,
    parameter int unsigned C = 1,
    parameter int unsigned PE = 1,
    parameter int unsigned NUM_STEPS = 1,
    parameter bit SIGNED = 1,
    parameter bit BASE_SIGNED = 1,
    parameter bit STEP_SIGNED = 1,
    parameter int BIAS = 0,
    parameter BASE_PATH = "",
    parameter STEP_PATH = "",
    parameter ERROR_PATH = ""
    , parameter COUNT_PATH = ""
) (
    input logic clk,
    input logic rst,
    output logic irdy,
    input logic ivld,
    input logic [PE-1:0][WI-1:0] idat,
    input logic ordy,
    output logic ovld,
    output logic [PE-1:0][O_BITS-1:0] odat
);
    localparam int unsigned CF = C / PE;
    localparam int unsigned FOLD_BITS = CF <= 1 ? 1 : $clog2(CF);
    localparam int unsigned STEP_BITS = NUM_STEPS <= 1 ? 1 : $clog2(NUM_STEPS);
    localparam int unsigned COMP_W = (WT > WI ? WT : WI) + 1;

    logic [BW-1:0] base_mem [PE][CF];
    logic [SW-1:0] step_mem [PE][CF];
    logic error_mem [PE][CF * NUM_STEPS];
    logic [CW-1:0] count_mem [PE][CF];
    logic [PE-1:0][WI-1:0] input_reg;
    logic [PE-1:0][O_BITS-1:0] count_reg;
    logic [PE-1:0][EW-1:0] residual_reg;
    logic [FOLD_BITS-1:0] fold_reg;
    logic [STEP_BITS-1:0] step_index;
    logic busy;
    logic output_valid;
    logic [PE-1:0][O_BITS-1:0] output_reg;

    initial begin
        if (CF * PE != C) begin
            $error("Delta thresholding requires C to be divisible by PE.");
            $finish;
        end
    end

    for (genvar pe = 0; pe < PE; pe++) begin : init_mem
        initial begin
            if (BASE_PATH != "")
                $readmemh($sformatf("%s%0d.dat", BASE_PATH, pe), base_mem[pe]);
            if (STEP_PATH != "")
                $readmemh($sformatf("%s%0d.dat", STEP_PATH, pe), step_mem[pe]);
            if (ERROR_PATH != "")
                $readmemh($sformatf("%s%0d.dat", ERROR_PATH, pe), error_mem[pe]);
            if (COUNT_PATH != "")
                $readmemh($sformatf("%s%0d.dat", COUNT_PATH, pe), count_mem[pe]);
        end
    end

    assign irdy = !busy && (!output_valid || ordy);
    assign ovld = output_valid;
    assign odat = output_reg;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            output_valid <= 1'b0;
            fold_reg <= '0;
            step_index <= '0;
            input_reg <= '0;
            count_reg <= '0;
            residual_reg <= '0;
            output_reg <= '0;
        end else begin
            if (output_valid && ordy)
                output_valid <= 1'b0;

            if (!busy && irdy && ivld) begin
                input_reg <= idat;
                count_reg <= '0;
                residual_reg <= '0;
                step_index <= '0;
                busy <= 1'b1;
                if (CF > 1) begin
                    if (fold_reg == FOLD_BITS'(CF - 1))
                        fold_reg <= '0;
                    else
                        fold_reg <= fold_reg + 1'b1;
                end
            end else if (busy) begin
                for (int pe = 0; pe < PE; pe++) begin
                    logic signed [COMP_W-1:0] base_value;
                    logic signed [COMP_W-1:0] step_value;
                    logic signed [COMP_W-1:0] threshold_value;
                    logic signed [COMP_W-1:0] input_value;
                    logic [EW-1:0] residual_value;
                    logic comparison;
                    logic [CW-1:0] count_value;
                    integer error_index;
                    integer output_value;
                    if (BASE_SIGNED)
                        base_value = $signed(
                            {{(COMP_W-BW){base_mem[pe][fold_reg][BW-1]}}, base_mem[pe][fold_reg]}
                        );
                    else
                        base_value = $signed(
                            {{(COMP_W-BW){1'b0}}, base_mem[pe][fold_reg]}
                        );
                    if (STEP_SIGNED)
                        step_value = $signed(
                            {{(COMP_W-SW){step_mem[pe][fold_reg][SW-1]}}, step_mem[pe][fold_reg]}
                        );
                    else
                        step_value = $signed(
                            {{(COMP_W-SW){1'b0}}, step_mem[pe][fold_reg]}
                        );
                    error_index = int'(fold_reg) * NUM_STEPS + int'(step_index);
                    residual_value = residual_reg[pe] + error_mem[pe][error_index];
                    count_value = count_mem[pe][fold_reg];
                    threshold_value = base_value
                        + step_value * $signed({1'b0, step_index})
                        + residual_value;
                    if (SIGNED)
                        input_value = $signed(
                            {{(COMP_W-WI){input_reg[pe][WI-1]}}, input_reg[pe]}
                        );
                    else
                        input_value = $signed({{(COMP_W-WI){1'b0}}, input_reg[pe]});
                    comparison = (step_index < count_value) && (threshold_value <= input_value);
                    count_reg[pe] <= count_reg[pe] + comparison;
                    residual_reg[pe] <= residual_value;
                    output_value = int'(count_reg[pe]) + int'(comparison) + BIAS;
                    if (step_index == STEP_BITS'(NUM_STEPS - 1))
                        output_reg[pe] <= output_value[O_BITS-1:0];
                end

                if (step_index == STEP_BITS'(NUM_STEPS - 1)) begin
                    busy <= 1'b0;
                    output_valid <= 1'b1;
                end else begin
                    step_index <= step_index + 1'b1;
                end
            end
        end
    end
endmodule