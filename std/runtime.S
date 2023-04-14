# To make a system call in 64-bit Linux, place the system call number in rax,
# and its arguments, in order, in rdi, rsi, rdx, r10, r8, and r9, then invoke
# syscall. Return values are stored in rax.

# If the callee wishes to use registers rbx, rsp, rbp, and r12–r15, it must
# restore their original values before returning control to the caller. All
# other registers must be saved by the caller if it wishes to preserve their
# values.

# All 64-bit registers (including rbp) require a REX prefix, which increases
# code size. Therefore, 32-bit registers (like ebp) are preferred where
# possible. Note that 32-bit values are zero-extended before they are written to
# a register (although that's not the case with 8-bit and 16-bit values).

.section .text

.global _start
_start:
    # Terminate the stack frame linked list; useful when compiling with
    # -fno-omit-frame-pointer.
    xor     %ebp, %ebp
    push    %rbp                    # rip = 0
    push    %rbp                    # rbp = 0
    mov     %rsp, %rbp

    # System V AMD64 ABI dictates that the first two arguments are stored in
    # rdi and rsi. The Graphene main() should use the same convention.
    call main

    # exit(int status)
    mov     %eax, %edi              # Program return code.
    mov     $60, %eax
    syscall

# https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf
# Userspace applications use the following register sequence
#   %rdi, %rsi, %rdx, %rcx, %r8 and %r9  (then stack)
# The kernel uses the following register sequence (clobbering %rcx and %r11)
#   %rdi, %rsi, %rdx, %r10, %r8 and %r9  (with %rax as the syscall number)
#
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
