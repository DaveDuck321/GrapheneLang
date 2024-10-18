@require_once "std/volatile.c3"
@require_once "std/assert.c3"

function [T] test_accessor : (storage: int mut&, accessor: T) -> void = {
    storage = 7;
    runtime_assert(accessor:read() == storage);

    accessor:write(8);
    runtime_assert(storage == 8);
    runtime_assert(accessor:read() == storage);

    storage == 33;
    runtime_assert(accessor:read() == storage);
}

function main : () -> int = {
    mut data_1 : int = 1;

    // Its kinda hard to test that the volatile accessors are actually truly volatile
    // Let's just check that there's not an obvious logic error somewhere.

    // TODO: once we have private functions we'll generate much less IR so maybe we
    // could setup an appropriate IR regex. 
    test_accessor(&mut data_1, make<Volatile<int>>(&mut data_1));
    test_accessor(&mut data_1, make<ReadVolatile<int>>(&mut data_1));
    test_accessor(&mut data_1, make<WriteVolatile<int>>(&mut data_1));
    test_accessor(&mut data_1, make<NonVolatile<int>>(&mut data_1));

    return 0;
}

/// @COMPILE
/// @RUN
