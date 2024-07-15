.global _start

_start:
    di
    ld sp, 0xcfff
    ld a, 0x00
    ld hl, 0x0000
    call main
    trap
