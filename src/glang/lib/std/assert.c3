@require_once "syscalls.c3"
@require_once "signal.c3"

function runtime_assert : (condition : bool) -> void = {
    if !condition {
        sys_assert_failed();
    }
}
