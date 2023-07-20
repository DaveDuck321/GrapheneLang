# Object files built from raw assembly, have an executable stack by default.
# See https://www.redhat.com/en/blog/linkers-warnings-about-executable-stacks-and-segments.
.section .note.GNU-stack, "", %progbits

.section .text

# A subroutine must preserve the contents of the registers r4-r8, r10, r11 and
# SP (and r9 in PCS variants that designate r9 as v6). [aapcs32]

.global _start
_start:
    # The first value on the stack is the argument count (argc). Pop it and
    # place it into r0, the register that stores the first argument to be
    # passed into main().
    # Next on the stack is the argument pointers array (argv). Set r1, the
    # register that stores the second argument of main(), to its address.
    # This is not part of the EABI, but it matches the SystemV AMD64 ABI.
    pop     {r0}
    mov     r1, sp

    # TODO 6.2.1.4 The Frame Pointer

    # The stack must be double-word aligned at a public interface. [aapcs32]
    and     sp, -0x8
    mov     fp, sp

    # Call main.
    bl      main

    # exit(int status)
    mov     r7, 1
    swi     0

# On arm/EABI:
# - pass arguments with r0-r6
# - pass the system call number on r7
# - do the syscall with `swi 0`
# - the two return values are on r0 and r1
#
# However, user space only uses r0-r3 to pass arguments. Any other arguments are
# pushed on the stack (in order, so 4th argument first, then 5th argument below,
# etc). Therefore, the syscall helpers need to save r4-r7 first and then pop
# the rest of the arguments from the stack.

.global _syscall_1
_syscall_1:
    # Syscall number is passed via r7, but r7 must be preserved, so save it on
    # the stack.
    push    {r7}
    mov     r7, r1
    swi     0
    pop     {r7}
    bx      lr

.global _syscall_3
_syscall_3:
    # Syscall number is passed via r7, but r7 must be preserved, so save it on
    # the stack.
    push    {r7}
    mov     r7, r3
    swi     0
    pop     {r7}
    bx      lr
