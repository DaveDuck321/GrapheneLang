@require_once "std/arithmetic.c3"
@require_once "std/vector.c3"
@require_once "std/format.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();

    mut vec : Vector<int> = make<Vector<int>>(&mut allocator);
    vec:push_back(0);
    vec:push_back(1);
    vec:push_back(2);
    vec:push_back(3);
    vec:push_back(4);
    vec:push_back(5);

    if vec:pop_back() != 5 {
        return 1;
    }

    if vec:get(1) != 1 {
        return 2;
    }

    vec[0] = 10;
    if vec:get(0) != 10 {
        return 3;
    }

    mut sum : int = 0;
    for i in vec:get_iter() {
        sum += i;
    }
    if sum != 20 {
        return sum;
    }

    vec:deallocate(); // TODO: RAII
    return 0;
}

/// @COMPILE
/// @RUN
