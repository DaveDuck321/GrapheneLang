@require_once "std/format.c3"
@require_once "std/rb_tree.c3"
@require_once "std/string.c3"

function main : () -> int = {
    let tree : RB_Tree<int, StringView> = make<RB_Tree<int, StringView>>();

    tree:insert(0, sv("V0"));
    tree:insert(1, sv("V1"));
    tree:insert(2, sv("V2"));
    tree:insert(3, sv("V3"));

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
/// V3
/// V2
/// V1
/// V0
