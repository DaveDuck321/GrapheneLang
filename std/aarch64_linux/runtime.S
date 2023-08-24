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
    # The first value on the stack is the argument count (argc). Load it into
    # r0, the register that stores the first argument to be passed into main().
    # Next on the stack is the argument pointers array (argv). Set r1, the
    # register that stores the second argument of main(), to its address.
    # NOTE the stack is (presumably!) already aligned to a 16-byte boundary at
    # this point. If we deallocate argc (i.e. set sp = sp + 8), then the stack
    # will no longer be aligned. If we then try to re-align the stack, we'll end
    # up re-allocating those 8 bytes. Therefore, don't bother popping argc off
    # the stack.
    ldr     x0, [sp]
    add     x1, sp, #8

    # Zero the frame pointer and the link register to indicate the end of the
    # frame record chain. Technically, we are supposed to push these on the
    # stack, but the prologue of main can do that if it wants too (clang omits
    # frame pointers by default, so only the link register would be saved).
    # https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#646the-frame-pointer
    mov     lr, xzr
    mov     fp, xzr

    # lli thinks it called main(), so argc and argv are already in r0 and r1,
    # respectively. Thus, skip the code that retrieves them from the stack.
.global _lli_start
_lli_start:

    # Call main.
    bl      main

    # exit(int status)
    mov     w8, #93
    svc     #0

# For aarch64 on Linux:
# - pass arguments with x0-x5
# - pass the system call number on w8
# - do the syscall with `svc #0`
# - the two return values are on x0 and x1
#
# User space passes the first eight arguments using registers r0-r7. Therefore,
# there are sufficient registers for all possible syscalls, and the helpers only
# need to move the syscall number into w8. Register r8 is not callee-saved, so
# we should be able to just overwrite it.

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
MAKE_SYSCALL(4, w4)
MAKE_SYSCALL(6, w6)
