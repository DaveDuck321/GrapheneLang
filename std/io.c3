@require_once "syscalls.c3";

function[Len] puts : (str: u8[Len]&) -> int = {
    // TODO reimplement this using only one syscall.
    sys_write(/* stdout */ 1, &str, Len);
    sys_write(/* stdout */ 1, "\n", 1);

    return 0;
};
