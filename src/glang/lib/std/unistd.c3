@require_once "syscalls.c3"
@require_once "io.c3"
@require_once "util.c3"
@require_once "sys/types.c3"

typedef Pipe : {
    read : File,
    write : File,
}

function fork : () -> pid_t = {
    return Narrow<pid_t>(sys_fork());
}

function ftruncate : (fd : int, length : isize) -> int = {
    return Narrow<int>(sys_ftruncate(fd, length));
}

function pipe : () -> Optional<Pipe> = {
    mut fds : int[2] = { 0, 0 };
    if sys_pipe(&fds) == 0 {
        return make<Optional<Pipe>>({ { fds[0] }, { fds[1] } });
    }

    return make<Optional<Pipe>>();
}

function getpid : () -> pid_t = {
    return sys_getpid();
}
