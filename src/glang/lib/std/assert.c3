@require_once "syscalls.c3"
@require_once "signal.c3"

function runtime_assert : (condition : bool) -> void = {
    if !condition {
        sys_write(/* stderr */ 2, "Assertion Failed!\n", 18);
        // Looks like lli catches SIGABRT and then resumes program execution.
        // As a result, tests that should fail... pass. We still want to raise
        // SIGABRT, as we can (a) catch it in gdb and (b) it creates a core
        // dump, so raise SIGKILL /after/ SIGABRT to make sure the process
        // terminates even under lli.
        raise(SIGABRT());
        raise(SIGKILL());
    }
}
