@require_once "unistd.c3"
@require_once "sys/types.c3"

typedef Process : {
    // TODO other 2 streams.
    pid : pid_t,
    // stdin : File,
    stdout : File,
    // stderr : File,
}

function execute : (
    path : CString, args : View<CString>
) -> Optional<Process> = {
    mut argv : CString[32];
    argv[0] = path;
    mut index : int = 1;
    for arg in args:get_iter() {
        argv[index] = arg;
        // FIXME ++index also compiles; but results in broken code?
        index += 1;
    }
    argv[index] = { &__builtin_int_to_ptr<u8[&]>(0) };

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
        // FIXME this reinterpret cast is gross.
        sys_execve(&path.data, &reinterpret_addr<u8[&][&]>(&argv));
        sys_exit(1);
    }

    // Parent process.
    close(&stdout:data().write);
    return make<Optional<Process>>({ pid, stdout:data().read });
}
