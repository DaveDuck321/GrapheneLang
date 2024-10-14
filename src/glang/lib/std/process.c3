@require_once "unistd.c3"
@require_once "sys/types.c3"
@require_once "sys/wait.c3"

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
    // TODO replace this with a vector when we have a global allocator.
    mut argv : CString[32];
    argv[0] = path;
    mut index : int = 1;
    for arg in args:get_iter() {
        argv[index] = arg;
        // FIXME ++index also compiles; but results in broken code?
        index += 1;
    }
    let null : iptr = 0;
    argv[index] = { &addr_to_heap_array<u8>(null) };

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

function wait : (process : Process mut&) -> Optional<int> = {
    if process.pid < 0 {
        return make<Optional<int>>();
    }

    // let status : Optional<int> = waitpid(process.pid, WEXITED());
    let status : Optional<int> = waitpid(process.pid, 0);
    if status:has_value() {
        process.pid = -1;
    }

    return status;
}

function close : (process : Process mut&) -> void = {
    process:wait();
    process.stdout:close();
}
