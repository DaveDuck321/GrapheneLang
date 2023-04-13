@require_once "io.c3";
@require_once "logical.c3";

function runtime_assert : (condition : bool) -> void = {
    if (!condition) {
        puts<17>("Assertion Failed!");
        sys_exit(1);
    }
};

function [T] sizeof : () -> isize = {
    return __builtin_sizeof<T>();
};

function [T] alignof : () -> isize = {
    return __builtin_alignof<T>();
};

function [T] addr_to_ref : (bad_address : i32) -> T& = {
    // NOTE: this overload ensures that we are called with an i64 (with no implict conversions)
    //       ideally we could prevent this with SFINAE (using partial specialization)

    // TODO: this should be a static assert/ error
    runtime_assert(false);
    return &__builtin_int_to_ptr<T&>(0);
};

function [T] addr_to_ref : (address : iptr) -> T& = {
    return &__builtin_int_to_ptr<T&>(address);
};
