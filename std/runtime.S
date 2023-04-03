.global _start
.global _syscall_0
.global _syscall_3

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

_syscall_0:
    # Syscall with 0 arguments. Syscall number should be in rdi.
    mov     %rdi, %rax
    syscall
    ret

_syscall_3:
    # Syscall with 3 arguments. rdi, rsi, and rdx are passed through to the
    # kernel. Syscall number should be in rcx.
    mov     %rcx, %rax
    syscall
    ret
