@require_once "std/arithmetic.c3"
@require_once "std/vector.c3"


function main : () -> int = {
    let allocator : Allocator = initialize_allocator();

    let vec : Vector<int> = make_vector<int>(&allocator);
    vec:push_back(0);
    vec:push_back(1);
    vec:push_back(2);
    vec:push_back(3);
    vec:push_back(4);
    vec:push_back(5);

    if (vec:pop_back() != 5) {
        return 1;
    }

    if (vec:get(1) != 1) {
        return 1;
    }

    vec:set(0, 10);
    if (vec:get(0) != 10) {
        return 1;
    }

    vec:deallocate(); // TODO: RAII
    return 0;
}

/// @COMPILE
/// @RUN
