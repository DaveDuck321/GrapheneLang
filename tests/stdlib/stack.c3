@require_once "std/arithmetic.c3"
@require_once "std/stack.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();

    mut stack : Stack<int> = make<Stack<int>>(&mut allocator);

    stack:push(0);
    stack:push(1);
    stack:push(2);
    stack:push(3);
    stack:push(4);
    stack:push(5);

    runtime_assert(stack:top() == 5);
    runtime_assert(stack:pop() == 5);

    runtime_assert(stack:top() == 4);
    runtime_assert(stack:length() == 5);
    runtime_assert(!stack:empty());

    stack:pop();
    stack:pop();
    stack:pop();
    stack:pop();
    stack:pop();

    runtime_assert(stack:length() == 0);
    runtime_assert(stack:empty());

    stack:push(10);
    stack:push(20);

    runtime_assert(stack:top() == 20);

    runtime_assert(stack:length() == 2);

    stack:deallocate();

    return 0;
}

/// @COMPILE
/// @RUN
