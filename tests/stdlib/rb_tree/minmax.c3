@require_once "std/rb_tree.c3"
@require_once "std/iterators.c3"

function main : () -> int = {
    mut allocator : Allocator = initialize_allocator();
    mut tree : RB_Tree<int> = make<RB_Tree<int>>(&mut allocator);
    mut res : Optional<int>;

    for i in range(-10, 11) {
        tree:insert(i);
    }

    res = tree:minimum();
    if res:data() != -10 {
        return 1;
    }

    res = tree:maximum();
    if res:data() != 10 {
        return 2;
    }

    return 0;
}

/// @COMPILE
/// @RUN
