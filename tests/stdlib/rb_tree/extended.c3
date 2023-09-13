@require_once "std/rb_tree.c3"
@require_once "std/iterators.c3"
@require_once "std/wrappers.c3"

function [K, V] is_valid_rb_tree_impl : (node : RB_Node<K, V>&) -> Optional<int> = {
    if node.colour == RED() {
        // Property 4: if a node is red, both of its children are black.
        if node.left:is_red() or node.right:is_red() {
            return make_optional<int>();
        }
    }

    // Property 3: every leaf node (nil node) is black.
    // Property 5: starting from any node, all simple paths down to leaf nodes
    // hold the same number of black nodes.
    let black_count_l : int = 1;
    let black_count_r : int = 1;

    if node.left:has_value() {
        let left_res : Optional<int> = node.left:data():is_valid_rb_tree_impl();
        if left_res:has_value() {
            black_count_l = left_res:data();
        }
        else {
            return make_optional<int>();
        }
    }
    if node.right:has_value() {
        let right_res : Optional<int> = node.right:data():is_valid_rb_tree_impl();
        if right_res:has_value() {
            black_count_r = right_res:data();
        }
        else {
            return make_optional<int>();
        }
    }

    if black_count_l == black_count_r {
        if node.colour == BLACK() {
            black_count_l += 1;
        }
        return make_optional(black_count_l);
    }

    return make_optional<int>();
}

function [K, V] is_valid_rb_tree : (tree : RB_Tree<K, V>&) -> bool = {
    // Property 2: the root node is black.
    if tree.root:is_red() {
        return false;
    }

    if tree.root:has_value() {
        let ret : Optional<int> = tree.root:data():is_valid_rb_tree_impl();
        return ret:has_value();
    }

    return true;
}

function main : () -> int = {
    let tree : RB_Tree<int, int> = make<RB_Tree<int, int>>();

    for i in range(10000) {
        tree:insert(i, i);
        if !tree:is_valid_rb_tree() {
            return i;
        }
    }

    for i in range(10000) {
        let value : Optional<int> = tree:search(i);

        if !value:has_value() or value:data() != i {
            return i;
        }
    }

    for i in range(1, 1000) {
        let value : Optional<int> = tree:search(-i);

        if value:has_value() {
            return -i;
        }
    }

    return 0;
}

/// @COMPILE
/// @RUN
