// sys/wait.h - declarations for waiting
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/wait.h.html

@require_once "types.c3"

function WEXITSTATUS : (status : int) -> int = {
    // If WIFEXITED(status), the low-order 8 bits of the status.
    return as_arithmetic((as_logical(status) & 0xff00) >> 8);
}

function WTERMSIG : (status : int) -> int = {
    // If WIFSIGNALED(status), the terminating signal.
    return as_arithmetic(as_logical(status) & 0x7f);
}

function WIFEXITED : (status : int) -> bool = {
    // True if status indicates normal termination.
    return WTERMSIG(status) == 0;
}

function waitpid : (pid : pid_t, options : int) -> Optional<int> = {
    mut wstatus : int = 0;
    let ret : int = sys_wait4(pid, &mut wstatus, options, 0);

    if ret > 0 {
        return make<Optional<int>>(wstatus);
    }

    printf(sv("wait4 returned %"), ret);
    return make<Optional<int>>();
}
