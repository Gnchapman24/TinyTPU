# TinyTPU out-of-context synthesis of one module, with resource expectations.
#
# Windows (PowerShell):
#   .\syn\vivado.ps1 -Ooc -Top pe
# or directly:
#   vivado -mode batch -nojournal -log <out_dir>\vivado.log -source <repo>\syn\ooc.tcl -tclargs <top> <out_dir>
#
# Synthesizes <top> out of context (no I/O buffers) for the Basys3 part with a 100 MHz clock on
# `clk`, then:
#   * writes <top>_ooc_utilization.rpt and <top>_ooc.json (cell counts and DSP register settings),
#   * checks the expectations listed below for <top>, and fails (non-zero exit) on any mismatch.

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

if {![info exists sources($top)]} {
  error "unknown top '$top'; known: [array names sources]"
}

file mkdir $out
foreach m $sources($top) {
  read_verilog -sv [file join $root rtl $m.sv]
}
synth_design -top $top -part $part -mode out_of_context
create_clock -name clk -period 10.0 [get_ports clk]

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
  if {[string is integer -strict $v]} {
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
