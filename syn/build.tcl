# TinyTPU Vivado batch build: synthesis, implementation, reports, bitstream.
#
# Windows (PowerShell), from any directory:
#   vivado -mode batch -nojournal -log <out_dir>\vivado.log -source <repo>\syn\build.tcl -tclargs <top> <out_dir>
# or use syn\vivado.ps1, which fills in the paths. <out_dir> must be on a local Windows disk,
# never inside the repo (AGENTS.md §5a). Run Vivado from <out_dir> so its scratch files land there.
#
# Writes to <out_dir>: the routed checkpoint (<top>_routed.dcp); reports (utilization, timing
# summary, the 20 worst setup paths, power, DRC); and <top>_metrics.json (Slice LUTs, LUT as memory, FFs, DSP,
# BRAM tiles, WNS / WHS, power), which tinytpu.baseline compares with syn/baseline.json.
# Fails (non-zero exit, no bitstream) if timing is not met or, for tinytpu_top, a resource
# ceiling of AGENTS.md is exceeded (LUT <= 5200, FF <= 6240, BRAM <= 40, DSP exactly 66).

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

write_checkpoint -force [file join $out ${top}_routed.dcp]
report_utilization -file [file join $out ${top}_utilization.rpt]
report_utilization -hierarchical -file [file join $out ${top}_utilization_hier.rpt]
report_timing_summary -file [file join $out ${top}_timing.rpt]
report_timing -setup -max_paths 20 -nworst 1 -unique_pins -file [file join $out ${top}_paths.rpt]
report_drc -file [file join $out ${top}_drc.rpt]

report_power -file [file join $out ${top}_power.rpt]

# Metrics (A5.2). DSP and BRAM are counted like syn/ooc.tcl (leaf cells by REF_NAME; BRAM in
# tiles, RAMB36 = 1, RAMB18 = 1/2). Power is Vivado's vectorless estimate: compare versions with it, nothing more.
proc count_cells {pattern} {
  return [llength [get_cells -hierarchical -quiet -filter "IS_PRIMITIVE && REF_NAME =~ $pattern"]]
}
# LUT and FF as the device sees them (report_utilization: "Slice LUTs", which packs two small
# LUT cells into one LUT6 site, and "Slice Registers"); the ceilings are device percentages.
set util [report_utilization -return_string]
if {![regexp {\|\s*Slice LUTs\*?\s*\|\s*([0-9]+)} $util -> metric(lut)] ||
    ![regexp {\|\s*LUT as Memory\s*\|\s*([0-9]+)} $util -> metric(lut_memory)] ||
    ![regexp {\|\s*Slice Registers\s*\|\s*([0-9]+)} $util -> metric(ff)]} {
  error "could not read the LUT / FF counts from report_utilization"
}
set metric(lut_cells) [count_cells "LUT*"]    ;# informational: LUT cells before packing
set metric(dsp)      [count_cells "DSP48E1"]
set metric(bram)     [format %g [expr {[count_cells "RAMB36*"] + [count_cells "RAMB18*"] / 2.0}]]
set metric(wns_ns)   [format %.3f [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]]
set metric(whs_ns)   [format %.3f [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]]
set power [report_power -return_string]
if {![regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $power -> metric(power_total_w)] ||
    ![regexp {Dynamic \(W\)\s*\|\s*([0-9.]+)} $power -> metric(power_dynamic_w)]} {
  error "could not read the power totals from report_power"
}
set fh [open [file join $out ${top}_metrics.json] w]
puts $fh "\{"
puts $fh "  \"top\": \"$top\","
puts $fh "  \"part\": \"$part\","
puts $fh "  \"vivado\": \"[version -short]\","
set keys [lsort [array names metric]]
for {set i 0} {$i < [llength $keys]} {incr i} {
  set k [lindex $keys $i]
  puts $fh "  \"$k\": $metric($k)[expr {$i < [llength $keys] - 1 ? "," : ""}]"
}
puts $fh "\}"
close $fh
puts "TinyTPU metrics: [file join $out ${top}_metrics.json]"
foreach k $keys { puts [format "  %-16s %s" $k $metric($k)] }

# Checks: the AGENTS.md hard ceilings (tinytpu_top only), then timing. No bitstream on failure.
set ceiling(tinytpu_top) {lut 5200 ff 6240 bram 40}
set exact(tinytpu_top)   {dsp 66}
set failures {}
if {[info exists ceiling($top)]} {
  foreach {k v} $ceiling($top) {
    if {$metric($k) > $v} { lappend failures "$k = $metric($k), above the ceiling $v" }
  }
}
if {[info exists exact($top)]} {
  foreach {k v} $exact($top) {
    if {$metric($k) != $v} { lappend failures "$k = $metric($k), must be exactly $v" }
  }
}
puts "TinyTPU timing: WNS = $metric(wns_ns) ns, WHS = $metric(whs_ns) ns"
if {$metric(wns_ns) < 0 || $metric(whs_ns) < 0} { lappend failures "timing not met (WNS $metric(wns_ns), WHS $metric(whs_ns))" }
if {[llength $failures]} {
  error "TinyTPU build $top: [join $failures {; }]"
}

write_bitstream -force [file join $out ${top}.bit]
puts "TinyTPU bitstream: [file join $out ${top}.bit]"
