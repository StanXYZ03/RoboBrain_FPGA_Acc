/*
 * Volatile ARM64 entry probe for headless ZCU102 diagnosis.
 *
 * Loaded at physical 0x14000000. The Image first instruction is temporarily
 * replaced by a branch here. Preserve x0 (DTB pointer), x1-x3, and the Image's
 * original MZ-compatible instruction; write "KENT" to the reserved mailbox;
 * then jump to the Image's original branch target at 0x19106484.
 */

.section .text
.global _start
_start:
    movz x9, #0x0000
    movk x9, #0x7ff0, lsl #16
    movz w10, #0x4e54
    movk w10, #0x4b45, lsl #16
    str w10, [x9]
    dsb sy
    isb
    .word 0xfa405a4d
    movz x9, #0x6484
    movk x9, #0x1910, lsl #16
    br x9
