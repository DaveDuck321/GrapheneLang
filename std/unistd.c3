@require_once "std/syscalls.c3";
@require_once "std/util.c3";

typedef pid_t : int;

function fork : () -> pid_t = {
    return Narrow<pid_t>(sys_fork());
};
