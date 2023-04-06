@require_once "std/syscalls.c3";

typedef pid_t : int;

function fork : () -> pid_t = {
    return __builtin_narrow<pid_t>(sys_fork());
};
