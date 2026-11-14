# Program the Basys3 over JTAG with a bitstream (volatile: lost at power-off).
#
# Windows (PowerShell):
#   vivado -mode batch -nojournal -nolog -source <repo>\syn\program.tcl -tclargs <file.bit>
# or use syn\vivado.ps1 -Program.

if {[llength $argv] != 1} {
  error "usage: -tclargs <file.bit>"
}
set bit [file normalize [lindex $argv 0]]
if {![file exists $bit]} {
  error "bitstream not found: $bit"
}

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7a35t*] 0]
if {$dev eq ""} {
  error "no xc7a35t found on the JTAG chain (is the Basys3 connected and powered?)"
}
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "TinyTPU: programmed $dev with $bit"
close_hw_target
disconnect_hw_server
close_hw_manager
