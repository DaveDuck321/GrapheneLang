@require_once "unistd.c3"
@require_once "sys/types.c3"

typedef Process : {
    // TODO other 2 streams.
    pid : pid_t,
    // stdin : File,
    stdout : File,
    // stderr : File,
}

function execute : (path : CString) -> Optional<Process> = {
    let stdout : Optional<Pipe> = pipe();
    if !stdout:has_value() {
        return make<Optional<Process>>();
    }

    let pid : pid_t = fork();
    if pid < 0 {
        return make<Optional<Process>>();
    }

    // Child process.
    if pid == 0 {
        close(&stdout:data().read);
        sys_dup2(stdout:data().write.fd, /* stdout */ 1);
        // TODO check return code... but how do we return the error?
        sys_execve(&path.data);
        sys_exit(1);
    }

    // Parent process.
    close(&stdout:data().write);
    return make<Optional<Process>>({ pid, stdout:data().read });
}
