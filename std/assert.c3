@require_once "syscalls.c3"

function runtime_assert : (condition : bool) -> void = {
    if !condition {
        sys_write(/* stderr */ 2, "Assertion Failed!", 17);
        sys_exit(1);
    }
}
