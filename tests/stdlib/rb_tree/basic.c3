@require_once "std/rb_tree.c3"
@require_once "std/format.c3"
@require_once "std/assert.c3"

function main : () -> int = {
    mut tree : RB_Tree<int> = make<RB_Tree<int>>();

    tree:insert(0);
    tree:insert(1);
    tree:insert(2);
    tree:insert(3);

    print(tree:search(4));
    print(tree:search(3));
    print(tree:search(2));
    print(tree:search(1));
    print(tree:search(0));

    runtime_assert(tree:delete(0));
    runtime_assert(tree:delete(1));
    runtime_assert(tree:delete(2));
    runtime_assert(tree:delete(3));
    runtime_assert(!tree:delete(4));

    print(tree:search(4));
    print(tree:search(3));
    print(tree:search(2));
    print(tree:search(1));
    print(tree:search(0));

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// <empty>
/// 3
/// 2
/// 1
/// 0
/// <empty>
/// <empty>
/// <empty>
/// <empty>
/// <empty>
