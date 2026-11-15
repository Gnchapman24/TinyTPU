# TinyTPU out-of-context synthesis of one module, with resource expectations.
#
# Windows (PowerShell):
#   .\syn\vivado.ps1 -Ooc -Top pe
# or directly:
#   vivado -mode batch -nojournal -log <out_dir>\vivado.log -source <repo>\syn\ooc.tcl -tclargs <top> <out_dir>
#
# Synthesizes <top> out of context (no I/O buffers) for the Basys3 part with a 100 MHz clock on
# `clk`, then:
#   * writes <top>_ooc_utilization.rpt and <top>_ooc.json (cell counts, DSP register settings,
#     and the worst setup slack WNS in ns, a post-synthesis estimate; ports get zero input and
#     output delay, i.e. they are timed as if driven by / driving a register next to the module),
#   * checks the expectations listed below for <top>, and fails (non-zero exit) on any mismatch.
#
# <top> is a module name, or a named configuration of a module (e.g. wbuf = buffer_ram with the
# WBUF parameters), listed in `module` and `generics` below.

if {[llength $argv] != 2} {
  error "usage: -tclargs <top> <out_dir>"
}
set top  [lindex $argv 0]
set out  [file normalize [lindex $argv 1]]
set root [file normalize [file join [file dirname [info script]] ..]]
set part xc7a35tcpg236-1

# Sources per module, in compile order.
set sources(pe)             {pe}
set sources(systolic_array) {pe systolic_array}
set sources(acc_unit)       {isa_pkg acc_unit}
set sources(requant)        {requant}
set sources(sequencer)      {isa_pkg instr_decode range_check sequencer}
set sources(matmul_ctrl)    {isa_pkg sync_fifo matmul_ctrl}
set sources(requant_ctrl)   {isa_pkg requant requant_ctrl}
set sources(tpu_core)       {isa_pkg sync_fifo buffer_ram pe systolic_array acc_unit requant instr_decode range_check sequencer matmul_ctrl requant_ctrl tpu_core}
set sources(wbuf)           {buffer_ram}
set sources(abuf)           {buffer_ram}
set sources(ptab)           {buffer_ram}

# Named configurations: the module to synthesize and its parameters (N = 8, memory map defaults).
set module(wbuf)   buffer_ram
set generics(wbuf) {WIDTH=64 DEPTH=16384 LANES=1 RD_PIPE=2 TAG_W=4}
set module(abuf)   buffer_ram
set generics(abuf) {WIDTH=64 DEPTH=1024 LANES=8 RD_PIPE=1 TAG_W=1}
set module(ptab)   buffer_ram
set generics(ptab) {WIDTH=54 DEPTH=512 LANES=1 RD_PIPE=1 TAG_W=4}

# Expectations per module: <metric> <value>. Metrics: LUT, FF (fabric flip-flops), DSP, BRAM,
# and for each DSP48E1 the register settings AREG, BREG, MREG, PREG (all DSPs must match).
#   pe: exactly one DSP; activation (AREG), stationary weight (BREG), product (MREG) and sum
#       (PREG) all inside it; no LUTs. The activation register is duplicated: one copy is the
#       DSP's AREG, the other is the 9 fabric flip-flops that drive a_out to the next PE.
set expect(pe) {DSP 1 LUT 0 FF 9 BRAM 0 AREG 1 BREG 1 MREG 1 PREG 1}
#   systolic_array (N = 8): one DSP per PE (64); every column's partial sums use the dedicated
#       PCOUT->PCIN cascade (7 links per column = 56); weight, product and sum registers inside
#       every DSP. AREG is not checked: Vivado may pull one input-skew register into the DSP
#       (AREG = 2), which is fine.
set expect(systolic_array) {DSP 64 CASCADE 56 BRAM 0 BREG 1 MREG 1 PREG 1}
#   acc_unit (N = 8, DEPTH = 512): the 512 x 256-bit buffer in 4 BRAM tiles (memory map); the
#       adders in fabric, never in DSPs (the DSP budget is exactly 66).
set expect(acc_unit) {DSP 0 BRAM 4}
#   requant: the 32 x 17-bit product (acc x M) in exactly 2 DSPs joined by the cascade, with
#       product and sum registers inside both; rounding shift and clamp in fabric. AREG/BREG are
#       not checked: the second DSP uses 2 input registers to line up with the first.
set expect(requant) {DSP 2 CASCADE 1 BRAM 0 MREG 1 PREG 1}
#   wbuf / abuf / ptab (buffer_ram, memory map at N = 8): exactly the memory map's BRAM tiles,
#       no DSPs, setup met. WBUF's 32 RAMB36 need a read-data mux: RD_PIPE = 2 uses the BRAM
#       output register plus one fabric register after the mux (the path to watch once placed).
#       ABUF's byte-lane writes must use the BRAM's own write enables (still 2 tiles).
#   sequencer (with instr_decode and range_check): control only. The range check's products are
#       shift-and-add, so no DSP may appear (the DSP budget is exactly 66).
set expect(sequencer) {DSP 0 BRAM 0 SLACK_MET 1}
#   matmul_ctrl (N = 8): the tile-loop address generator; incremental adders only, no DSP.
set expect(matmul_ctrl) {DSP 0 BRAM 0 SLACK_MET 1}
#   requant_ctrl: the requantizer's 2 DSPs and nothing else.
set expect(requant_ctrl) {DSP 2 BRAM 0 SLACK_MET 1}
#   tpu_core (N = 8, memory map depths): the whole core without the host link. DSP exactly 66
#       (64 PEs + 2 requantizer), BRAM exactly 39 (WBUF 32, ABUF 2, ACC 4, PTAB 1; IMEM is LUT RAM).
set expect(tpu_core) {DSP 66 BRAM 39 SLACK_MET 1}
set expect(wbuf) {DSP 0 BRAM 32 SLACK_MET 1}
set expect(abuf) {DSP 0 BRAM 2 SLACK_MET 1}
set expect(ptab) {DSP 0 BRAM 1 SLACK_MET 1}

if {![info exists sources($top)]} {
  error "unknown top '$top'; known: [array names sources]"
}

file mkdir $out
foreach m $sources($top) {
  read_verilog -sv [file join $root rtl $m.sv]
}
set mod [expr {[info exists module($top)] ? $module($top) : $top}]
set gen  [expr {[info exists generics($top)] ? $generics($top) : {}}]
if {[llength $gen]} {
  synth_design -top $mod -part $part -mode out_of_context -generic $gen
} else {
  synth_design -top $mod -part $part -mode out_of_context
}
create_clock -name clk -period 10.0 [get_ports clk]
set_input_delay  0 -clock clk [get_ports -quiet -filter {DIRECTION == IN && NAME != clk}]
set_output_delay 0 -clock clk [get_ports -quiet -filter {DIRECTION == OUT}]

report_utilization -file [file join $out ${top}_ooc_utilization.rpt]

# Count leaf cells by library cell name (more stable across Vivado versions than PRIMITIVE_TYPE).
proc count_cells {pattern} {
  return [llength [get_cells -hierarchical -quiet -filter "IS_PRIMITIVE && REF_NAME =~ $pattern"]]
}
set metrics(LUT)  [count_cells "LUT*"]
set metrics(FF)   [count_cells "FD*"]
set metrics(DSP)  [count_cells "DSP48E1"]
# BRAM in tiles: a RAMB36 is one tile, a RAMB18 half of one.
set metrics(BRAM) [format %g [expr {[count_cells "RAMB36*"] + [count_cells "RAMB18*"] / 2.0}]]

# Worst setup slack over all paths, ports included (see set_input_delay / set_output_delay above).
set metrics(WNS) [format %.3f [get_property SLACK [get_timing_paths -max_paths 1 -setup]]]
set metrics(SLACK_MET) [expr {$metrics(WNS) >= 0 ? 1 : 0}]

set dsps [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]

# CASCADE: DSPs whose PCIN is driven by another DSP's PCOUT (the dedicated column cascade).
set metrics(CASCADE) 0
# The net crosses module boundaries, so follow all its segments down to leaf pins.
foreach d $dsps {
  set net [get_nets -quiet -segments -of_objects [get_pins -quiet "$d/PCIN\[0\]"]]
  if {$net ne "" && [llength [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT && REF_PIN_NAME =~ PCOUT*}]]} {
    incr metrics(CASCADE)
  }
}

foreach prop {AREG BREG MREG PREG} {
  if {[llength $dsps] == 0} {
    set metrics($prop) "none"
    continue
  }
  set vals [lsort -unique [get_property $prop $dsps]]
  set metrics($prop) [expr {[llength $vals] == 1 ? [lindex $vals 0] : "mixed:$vals"}]
}

# JSON summary.
set fh [open [file join $out ${top}_ooc.json] w]
puts $fh "{"
puts $fh "  \"top\": \"$top\","
puts $fh "  \"part\": \"$part\","
set keys [lsort [array names metrics]]
for {set i 0} {$i < [llength $keys]} {incr i} {
  set k [lindex $keys $i]
  set v $metrics($k)
  set sep [expr {$i < [llength $keys] - 1 ? "," : ""}]
  if {[string is double -strict $v]} {
    puts $fh "  \"$k\": $v$sep"
  } else {
    puts $fh "  \"$k\": \"$v\"$sep"
  }
}
puts $fh "}"
close $fh

puts "TinyTPU OOC $top:"
foreach k $keys {
  puts [format "  %-5s %s" $k $metrics($k)]
}

set failures {}
if {[info exists expect($top)]} {
  foreach {k v} $expect($top) {
    if {$metrics($k) ne $v} {
      lappend failures "$k = $metrics($k), expected $v"
    }
  }
}
if {[llength $failures]} {
  error "TinyTPU OOC $top: [join $failures {; }]"
}
puts "TinyTPU OOC $top: all expectations met"
