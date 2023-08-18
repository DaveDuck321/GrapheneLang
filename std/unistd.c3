@require_once "syscalls.c3"
@require_once "util.c3"
@require_once "sys/types.c3"

function fork : () -> pid_t = {
    return Narrow<pid_t>(sys_fork());
}
