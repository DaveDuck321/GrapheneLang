# If the callee wishes to use registers rbx, rsp, rbp, and r12–r15, it must
# restore their original values before returning control to the caller. All
# other registers must be saved by the caller if it wishes to preserve their
# values.

# All 64-bit registers (including rbp) require a REX prefix, which increases
# code size. Therefore, 32-bit registers (like ebp) are preferred where
# possible. Note that 32-bit values are zero-extended before they are written to
# a register (although that's not the case with 8-bit and 16-bit values).

# Object files built from raw assembly, have an executable stack by default.
# See https://www.redhat.com/en/blog/linkers-warnings-about-executable-stacks-and-segments.
.section .note.GNU-stack, "", %progbits

.section .text

.global _start
_start:
    # The the user code should mark the deepest stack frame by setting the
    # frame pointer to zero. [Section 3.4.1 of docs/abi.pdf]
    xor     %ebp, %ebp

    # The first value of the stack is the argument count (argc). Pop it and
    # place it into %rdi, the register that stores the first argument to be
    # passed into main().
    # Next on the stack is the argument pointers array (argv). Set %rsi, the
    # register that stores the second argument of main(), to its address.
    # See Section 3.4.1 of docs/abi.pdf for more.
    pop     %rdi
    mov     %rsp, %rsi

    # lli thinks it called main(), so argc and argv are already in %rdi and
    # %rsi, respectively. Thus, skip the code that retrieves them from the
    # stack.
.global _lli_start
_lli_start:

    # The stack needs to be 16 byte aligned immediately before the call
    # instruction is executed. Therefore, AND %rsp with a sign-extended 0xf0 to
    # clear the 4 LSBs.
    # NOTE if we start passing 256-bit (AVX) or 512-bit (AVX-512) packed data
    # types on the stack, then the stack needs to be 32 or 64 byte aligned,
    # respectively. Right now, the widest built-in type is i128.
    # See Section 3.2.2 of docs/abi.pdf for more.
    and     $-0x10, %rsp

    # System V AMD64 ABI dictates that the first two arguments are stored in
    # rdi and rsi. The Graphene main() should use the same convention.
    call main

    # exit(int status)
    mov     %eax, %edi              # Program return code.
    mov     $60, %eax
    syscall

# Userspace applications use the following register sequence
#   %rdi, %rsi, %rdx, %rcx, %r8 and %r9  (then stack)
# The kernel uses the following register sequence (clobbering %rcx and %r11)
#   %rdi, %rsi, %rdx, %r10, %r8 and %r9  (with %rax as the syscall number)
# See Section 3.2.3 of docs/abi.pdf for more.

# Helper exposing `syscall` as a foreign function
# Relies on the caller to use the System V AMD64 ABI calling convention
# We clobber %rcx, %r10, %r11 (ok since we don't need to preserve them)

#define MAKE_SYSCALL(arg_count, syscall_number_reg) \
    .global _syscall_ ## arg_count;                 \
    _syscall_ ## arg_count:                         \
        mov     %rcx,               %r10;           \
        mov     syscall_number_reg, %rax;           \
        syscall;                                    \
        ret

MAKE_SYSCALL(0, %rdi)
MAKE_SYSCALL(1, %rsi)
MAKE_SYSCALL(2, %rdx)
MAKE_SYSCALL(3, %rcx)
MAKE_SYSCALL(6, 8(%rsp))
