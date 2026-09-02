/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	AXI-Stream skid buffer: one cycle of latency, full throughput,
 *		no combinational path from ordy to irdy.
 *****************************************************************************/

module lutneuron_skid #(
	parameter integer DATA_WIDTH = 1
)(
	input  wire                    clk,
	input  wire                    rst,

	input  wire [DATA_WIDTH-1:0]   idat,
	input  wire                    ivld,
	output wire                    irdy,

	output wire [DATA_WIDTH-1:0]   odat,
	output wire                    ovld,
	input  wire                    ordy
);

	// main output register
	reg [DATA_WIDTH-1:0] ODat = {DATA_WIDTH{1'b0}};
	reg                  OVld = 1'b0;
	// overflow buffer, holds the beat accepted in the cycle irdy went low
	reg [DATA_WIDTH-1:0] BDat = {DATA_WIDTH{1'b0}};
	reg                  BVld = 1'b0;

	assign  irdy = !BVld;
	assign  odat = ODat;
	assign  ovld = OVld;

	wire  out_free = !OVld || ordy;

	always @(posedge clk) begin
		if(rst) begin
			OVld <= 1'b0;
			BVld <= 1'b0;
		end
		else begin
			if(out_free) begin
				if(BVld) begin
					ODat <= BDat;
					OVld <= 1'b1;
					BVld <= 1'b0;
				end
				else begin
					ODat <= idat;
					OVld <= ivld;
				end
			end
			else if(irdy && ivld) begin
				// output stalled: capture the incoming beat in the skid slot
				BDat <= idat;
				BVld <= 1'b1;
			end
		end
	end

endmodule : lutneuron_skid
