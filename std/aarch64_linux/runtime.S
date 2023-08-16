# Object files built from raw assembly, have an executable stack by default.
# See https://www.redhat.com/en/blog/linkers-warnings-about-executable-stacks-and-segments.
.section .note.GNU-stack, "", %progbits

.section .text

# Software developers creating platform-independent code are advised to avoid
# using r18 if at all possible.
# A subroutine invocation must preserve the contents of the registers r19-r29
# and SP. All 64 bits of each value stored in r19-r29 must be preserved, even
# when using the ILP32 data model.
# In all variants of the procedure call standard, registers r16, r17, r29 and
# r30 have special roles. In these roles they are labeled IP0, IP1, FP and LR
# when being used for holding addresses (that is, the special name implies
# accessing the register as a 64-bit entity).
# https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#611general-purpose-registers
#
# Therefore, only use registers r0-r15.

.global _start
_start:
    # The first value on the stack is the argument count (argc). Pop it and
    # place it into r0, the register that stores the first argument to be
    # passed into main().
    # Next on the stack is the argument pointers array (argv). Set r1, the
    # register that stores the second argument of main(), to its address.
    # This is not part of the EABI, but it matches the SystemV AMD64 ABI.
    # pop     {r0}
    # mov     r1, sp

    # lli thinks it called main(), so argc and argv are already in r0 and r1,
    # respectively. Thus, skip the code that retrieves them from the stack.
.global _lli_start
_lli_start:

    # TODO 6.2.1.4 The Frame Pointer

    # The stack must be double-word aligned at a public interface. [aapcs32]
    # FIXME
    # and     sp, sp, #-16
    # mov     fp, sp

    # Call main.
    bl      main

    # exit(int status)
    mov     w8, 93
    svc     #0

# For arm64 on Linux:
# - pass arguments with x0-x5
# - pass the system call number on w8
# - do the syscall with `svc #0`
# - the two return values are on x0 and x1
#
# FIXME all below
# However, user space only uses r0-r3 to pass arguments. Any other arguments are
# pushed on the stack (in order, so 4th argument first, then 5th argument below,
# etc). Therefore, the syscall helpers need to save r4-r7 first and then pop
# the rest of the arguments from the stack.

# Syscall number is passed via r7, but r7 must be preserved, so save it on the
# stack.
# This helper is only appropriate for syscalls with less than 3 arguments, as
# it assumes that the caller has not passed any arguments with the stack.

# FIXME we're implicitly using the red zone, but I don't think the EABI allows
# for that. 6.2.1.1 A process may only store data in the closed interval of the
# entire stack delimited by [SP, stack base - 1] (where SP is the value of
# register r13).

#define MAKE_SYSCALL(arg_count, syscall_number_reg) \
    .global _syscall_ ## arg_count;                 \
    _syscall_ ## arg_count:                         \
        mov     w8, syscall_number_reg;             \
        svc     #0;                                 \
        ret

MAKE_SYSCALL(0, w0)
MAKE_SYSCALL(1, w1)
MAKE_SYSCALL(2, w2)
MAKE_SYSCALL(3, w3)
MAKE_SYSCALL(6, w6)
