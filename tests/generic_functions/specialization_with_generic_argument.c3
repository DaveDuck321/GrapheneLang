@require_once "std/arithmetic.c3"
@require_once "std/util.c3"

typedef [T] Optional : { has_value : bool, value : T }
typedef [T] Optional<T&> : { ptr : iptr } // Nullptr optimization

function [T] make<Optional<T>> : () -> Optional<T> = {
    mut result : Optional<T>;
    result.has_value = false;
    return result;
}

function [T] make<Optional<T>> : (value : T) -> Optional<T> = {
    return {true, value};
}

function [T] has_value : (opt : Optional<T> mut&) -> bool = {
    return opt.has_value;
}

function [T] data : (opt : Optional<T> mut&) -> T mut& = {
    return &mut opt.value;
}

function [T] make<Optional<T&>> : () -> Optional<T&> = {
    return {0};
}

function [T] make<Optional<T&>> : (data : T mut&) -> Optional<T&> = {
    return {ref_to_addr(&data)};
}

function [T] has_value : (opt : Optional<T&> mut&) -> bool = {
    return opt.ptr != 0;
}

function [T] data : (opt : Optional<T&> mut&) -> T mut& = {
    return &mut addr_to_mut_ref<T>(opt.ptr);
}

function main: () -> int = {
    mut non_specialized : Optional<int> = make<Optional<int>>(8);
    non_specialized:data() = 7;
    if !non_specialized:has_value() or non_specialized:data() != 7 {
        return 1;
    }

    non_specialized = make<Optional<int>>();
    if non_specialized:has_value() {
        return 2;
    }

    mut data : int = 9;
    mut specialized : Optional<int&> = make<Optional<int&>>(&mut data);
    if !specialized:has_value() or specialized:data() != 9 {
        return 3;
    }
    specialized:data() = 8;
    if data != 8 {
        return 4;
    }

    specialized = make<Optional<int&>>();
    if specialized:has_value() {
        return 5;
    }
    return 0;
}

/// @COMPILE
/// @RUN
