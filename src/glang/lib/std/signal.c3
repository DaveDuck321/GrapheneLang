@require_once "sys/types.c3"
@require_once "syscalls.c3"

function SIGABRT : () -> int = {
    return 6;
}

function kill : (pid : pid_t, sig : int) -> int = {
    return sys_kill(pid, sig);
}

function raise : (sig : int) -> int = {
    // Send a signal to yourself. We need to specify the pid explicitly
    // unfortunately.
    // NOTE use the raw getpid syscall here to avoid circular dependencies.
    return kill(sys_getpid(), sig);
}
