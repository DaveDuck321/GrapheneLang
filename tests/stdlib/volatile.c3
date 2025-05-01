@require_once "std/arithmetic.c3"
@require_once "std/assert.c3"
@require_once "std/util.c3"
@require_once "std/volatile.c3"

function [T] test_accessor : (storage: int mut&, accessor: T) -> void = {
    storage = 7;
    runtime_assert(accessor:read() == storage);

    accessor:write(8);
    runtime_assert(storage == 8);
    runtime_assert(accessor:read() == storage);

    storage == 33;
    runtime_assert(accessor:read() == storage);
}

function [T] test_read_only_accessor : (storage: int mut&, accessor: T) -> void = {
    storage = 7;
    runtime_assert(accessor:read() == storage);

    storage == 33;
    runtime_assert(accessor:read() == storage);
}

function main : () -> int = {
    mut data : int = 1;

    // Its kinda hard to test that the volatile accessors are actually truly volatile
    // Let's just check that there's not an obvious logic error somewhere.

    // TODO: once we have private functions we'll generate much less IR so maybe we
    // could setup an appropriate IR regex.
    test_accessor(&mut data, make<Volatile<int>>(&mut data));
    test_accessor(&mut data, make<WriteVolatile<int>>(&mut data));
    test_read_only_accessor(&mut data, make<ReadOnlyVolatile<int>>(&data));

    test_accessor(&mut data, make<Volatile<int>>(ref_to_addr(&data)));
    test_accessor(&mut data, make<WriteVolatile<int>>(ref_to_addr(&data)));
    test_read_only_accessor(&mut data, make<ReadOnlyVolatile<int>>(ref_to_addr(&data)));

    return 0;
}

/// @COMPILE
/// @RUN
