@require_once "std/rb_tree.c3"
@require_once "std/iterators.c3"
@require_once "std/format.c3"

function main : () -> int = {
    mut tree : RB_Tree<int> = make<RB_Tree<int>>();

    for i in range(-10, 11) {
        tree:insert(i);
    }

    let min : Optional<RB_Node<int> mut&> = tree:search_impl(-10);
    if !min:has_value() or min:data().key != -10 {
        return 1;
    }

    let max : Optional<RB_Node<int> mut&> = tree:search_impl(10);
    if !max:has_value() or max:data().key != 10 {
        return 2;
    }

    let min_predecessor : Optional<RB_Node<int> mut&> = min:data():predecessor();
    if min_predecessor:has_value() {
        return 3;
    }

    let max_successor : Optional<RB_Node<int> mut&> = max:data():successor();
    if max_successor:has_value() {
        return 4;
    }

    let nine : Optional<RB_Node<int> mut&> = max:data():predecessor();
    if !nine:has_value() or nine:data().key != 9 {
        return 5;
    }

    let eight : Optional<RB_Node<int> mut&> = nine:data():predecessor();
    if !eight:has_value() or eight:data().key != 8 {
        return 6;
    }

    let seven : Optional<RB_Node<int> mut&> = eight:data():predecessor();
    if !seven:has_value() or seven:data().key != 7 {
        return 7;
    }

    if seven:data():successor() != eight {
        return 8;
    }

    if eight:data():successor() != nine {
        return 9;
    }

    if nine:data():successor() != max {
        return 10;
    }

    for i in tree:get_citer() {
        print(i);
    }

    return 0;
}

/// @COMPILE
/// @RUN; EXPECT OUT
/// -10
/// -9
/// -8
/// -7
/// -6
/// -5
/// -4
/// -3
/// -2
/// -1
/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// 10
