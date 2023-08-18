typedef [T, @Len] ArrayIter : {
    array : T[@Len]&,
    index : isize,
}

function [T, @Len] length : (array : T[@Len]&) -> isize = {
    return @Len;
}

function [T, @Len] get_iter : (array : T[@Len]&) -> ArrayIter<T, @Len> = {
    return {&array, 0};
}

@implicit [T, @Len] has_next : (array_iter : ArrayIter<T, @Len>&) -> bool = {
    return array_iter.index < @Len;
}

@implicit [T, @Len] get_next : (array_iter : ArrayIter<T, @Len>&) -> T& = {
    let result : T& = &array_iter.array[array_iter.index];
    array_iter.index += 1;
    return &result;
}


function [T] sizeof : () -> isize = {
    return __builtin_sizeof<T>();
}

function [T] alignof : () -> isize = {
    return __builtin_alignof<T>();
}

function [T] ref_to_addr : (ref : T&) -> iptr = {
    return __builtin_ptr_to_int(&ref);
}

function [T] addr_to_ref : (bad_address : i32) -> T& = {
    // NOTE: this overload ensures that we are called with an i64 (with no implict conversions)
    //       ideally we could prevent this with SFINAE (using partial specialization)

    // TODO: this should be a static assert/ error
    runtime_assert(false);
    return &__builtin_int_to_ptr<T&>(0);
}

function [T] addr_to_ref : (address : iptr) -> T& = {
    return &__builtin_int_to_ptr<T&>(address);
}

function [T] addr_to_heap_array : (bad_address : i32) -> T[&] = {
    // TODO: this should be a static assert/ error
    runtime_assert(false);
    return &__builtin_int_to_ptr<T[&]>(bad_address);
}

function [T] addr_to_heap_array : (address : iptr) -> T[&] = {
    return &__builtin_int_to_ptr<T[&]>(address);
}

function as_arithmetic : (value : u8)   -> i8   = { return __builtin_bitcast<i8>(value); }
function as_arithmetic : (value : u16)  -> i16  = { return __builtin_bitcast<i16>(value); }
function as_arithmetic : (value : u32)  -> i32  = { return __builtin_bitcast<i32>(value); }
function as_arithmetic : (value : u64)  -> i64  = { return __builtin_bitcast<i64>(value); }
function as_arithmetic : (value : u128) -> i128 = { return __builtin_bitcast<i128>(value); }

function as_logical : (value : i8)   -> u8   = { return __builtin_bitcast<u8>(value); }
function as_logical : (value : i16)  -> u16  = { return __builtin_bitcast<u16>(value); }
function as_logical : (value : i32)  -> u32  = { return __builtin_bitcast<u32>(value); }
function as_logical : (value : i64)  -> u64  = { return __builtin_bitcast<u64>(value); }
function as_logical : (value : i128) -> u128 = { return __builtin_bitcast<u128>(value); }

function [T] has_next : (iter : T&) -> bool = { return iter:__builtin_has_next(); }
