@require_once "io.c3";
@require_once "logical.c3";

function runtime_assert : (condition : bool) -> void = {
    if (!condition) {
        puts<17>("Assertion Failed!");
        sys_exit(1);
    }
};
