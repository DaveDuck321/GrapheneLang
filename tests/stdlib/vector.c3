@require_once "std/arithmetic.c3"
@require_once "std/vector.c3"
@require_once "std/format.c3"

@operator[T] [] : (self : Vector<T>&, index : isize) -> T mut& = {
    return &mut self.memory.data[index];
}

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();

    mut vec : Vector<int> = make<Vector<int>>(&mut allocator);
    vec:push_back(0);
    vec:push_back(1);
    vec:push_back(2);
    vec:push_back(3);
    vec:push_back(4);
    vec:push_back(5);

    runtime_assert(vec:pop_back() == 5);
    runtime_assert(vec:get(1) == 1);

    vec[0] = 10;
    runtime_assert(vec:get(0) == 10);

    mut sum : int = 0;
    for i in vec:get_iter() {
        sum += i;
    }
    runtime_assert(sum == 20);

    vec:resize(100, 2);

    sum = 0;
    for i in vec:get_iter() {
        sum += i;
    }
    runtime_assert(sum == 210);

    vec:resize(5, 2);

    sum = 0;
    for i in vec:get_iter() {
        sum += i;
    }
    runtime_assert(sum == 20);

    vec:deallocate(); // TODO: RAII
    return 0;
}

/// @COMPILE
/// @RUN
