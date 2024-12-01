@require_once "syscalls.c3"
@require_once "io.c3"
@require_once "util.c3"
@require_once "sys/types.c3"

typedef Pipe : {
    read : File,
    write : File,
}

function fork : () -> pid_t = {
    // The fork syscall is not available on all architectures. Emulate it with
    // clone instead. I believe we don't need to set any of the CLONE_ flags,
    // just the exit signal.
    return sys_clone(SIGCHLD(), 0, 0, 0, 0);
}

function ftruncate : (fd : int, length : isize) -> int = {
    return Narrow<int>(sys_ftruncate(fd, length));
}

function pipe : () -> Optional<Pipe> = {
    mut fds : int[2] = { 0, 0 };
    if sys_pipe2(&mut fds, /* no flags */ 0) == 0 {
        return make<Optional<Pipe>>({ { fds[0] }, { fds[1] } });
    }

    return make<Optional<Pipe>>();
}

function getpid : () -> pid_t = {
    return sys_getpid();
}
