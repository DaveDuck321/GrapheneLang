@require_once "syscalls.c3";
@require_once "sys/uio.c3";

function[Len] puts : (str: u8[Len]&) -> int = {
    let iov : iovec[2];

    // FIXME: we should be able to initialise iov directly.
    iov[0] = {&str, Len};
    iov[1] = {"\n", 1};

    sys_writev(/* stdout */ 1, &iov, 2);

    return 0;
};
