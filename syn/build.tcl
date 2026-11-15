# TinyTPU Vivado batch build: synthesis, implementation, reports, bitstream.
#
# Windows (PowerShell), from any directory:
#   vivado -mode batch -nojournal -log <out_dir>\vivado.log -source <repo>\syn\build.tcl -tclargs <top> <out_dir>
# or use syn\vivado.ps1, which fills in the paths. <out_dir> must be on a local Windows disk,
# never inside the repo (AGENTS.md §5a). Run Vivado from <out_dir> so its scratch files land there.
#
# Fails (non-zero exit) if timing is not met.

if {[llength $argv] != 2} {
  error "usage: -tclargs <top> <out_dir>"
}
set top  [lindex $argv 0]
set out  [file normalize [lindex $argv 1]]
set root [file normalize [file join [file dirname [info script]] ..]]
set part xc7a35tcpg236-1

# RTL files per top level, in compile order (packages first).
set sources(loopback_top) {reset_gen uart_rx uart_tx sync_fifo crc16_ccitt loopback_echo loopback_top}
set sources(tinytpu_top) {isa_pkg reset_gen uart_rx uart_tx sync_fifo crc16_ccitt buffer_ram pe systolic_array acc_unit requant instr_decode range_check sequencer matmul_ctrl requant_ctrl tpu_core cmd_engine tinytpu_top}

if {![info exists sources($top)]} {
  error "unknown top '$top'; known: [array names sources]"
}

file mkdir $out
puts "TinyTPU build: top=$top part=$part"
puts "  repo: $root"
puts "  out:  $out"

foreach m $sources($top) {
  read_verilog -sv [file join $root rtl $m.sv]
}
read_xdc [file join $root syn basys3.xdc]

synth_design -top $top -part $part
report_utilization -file [file join $out ${top}_synth_utilization.rpt]

opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $out ${top}_utilization.rpt]
report_utilization -hierarchical -file [file join $out ${top}_utilization_hier.rpt]
report_timing_summary -file [file join $out ${top}_timing.rpt]
report_drc -file [file join $out ${top}_drc.rpt]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "TinyTPU timing: WNS = $wns ns, WHS = $whs ns"
if {$wns < 0 || $whs < 0} {
  error "timing not met (WNS $wns, WHS $whs)"
}

write_bitstream -force [file join $out ${top}.bit]
puts "TinyTPU bitstream: [file join $out ${top}.bit]"
