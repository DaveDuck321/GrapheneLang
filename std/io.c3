@require_once "syscalls.c3";

function[Len] puts : (str: u8[Len]&) -> int = {
    sys_write(/* stdout */ 1, &str, Len);

    return 0;
};
