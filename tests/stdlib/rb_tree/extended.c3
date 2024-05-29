@require_once "std/iterators.c3"
@require_once "std/rb_tree.c3"
@require_once "std/util.c3"
@require_once "std/wrappers.c3"

function [K] child_is_valid_rb_tree: (
    node : RB_Node<K> mut&, side : bool
) -> Optional<int> = {
    let child_opt : Optional<RB_Node<K> mut&> = node:get_child(side);

    if !child_opt:has_value() {
        return make<Optional<int>>(1);
    }

    let child : RB_Node<K> mut& = &mut child_opt:data();

    // Check that the child's parent reference is correct.
    if !child.parent:has_value() or
        ref_to_addr(&child.parent:data()) != ref_to_addr(&node) {
        return make<Optional<int>>();
    }

    if side == LEFT() {
        if child.key >= node.key {
            return make<Optional<int>>();
        }
    }
    else {
        if child.key <= node.key {
            return make<Optional<int>>();
        }
    }

    return child:is_valid_rb_tree_impl();
}

function [K] is_valid_rb_tree_impl : (node : RB_Node<K> mut&) -> Optional<int> = {
    // Property 4: if a node is red, both of its children are black.
    if node.colour == RED() and (node.left:is_red() or node.right:is_red()) {
        return make<Optional<int>>();
    }

    // Property 3: every leaf node (nil node) is black.
    // Property 5: starting from any node, all simple paths down to leaf nodes
    // hold the same number of black nodes.
    let black_count_l : Optional<int> = node:child_is_valid_rb_tree(LEFT());
    if !black_count_l:has_value() {
        return make<Optional<int>>();
    }

    let black_count_r : Optional<int> = node:child_is_valid_rb_tree(RIGHT());
    if !black_count_r:has_value() {
        return make<Optional<int>>();
    }

    if black_count_l:data() != black_count_r:data() {
        return make<Optional<int>>();
    }

    if node.colour == BLACK() {
        return make<Optional<int>>(black_count_l:data() + 1);
    }

    return black_count_l;
}

function [K] is_valid_rb_tree : (tree : RB_Tree<K> mut&) -> bool = {
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
    mut allocator : Allocator = initialize_allocator();
    mut tree : RB_Tree<int> = make<RB_Tree<int>>(&mut allocator);
    mut n : int = 1000;

    for i in range(n) {
        tree:insert(i);
        if !tree:is_valid_rb_tree() {
            return 1;
        }
    }

    for i in range(n) {
        let value : Optional<int> = tree:search(i);
        if !value:has_value() or value:data() != i {
            return 2;
        }
    }

    for i in range(1, n) {
        let value : Optional<int> = tree:search(-i);
        if value:has_value() {
            return 3;
        }
    }

    for i in range(1, n, 5) {
        tree:delete(i);
        if !tree:is_valid_rb_tree() {
            return 4;
        }
    }

    for i in range(0, n, 7) {
        tree:delete(i);
        if !tree:is_valid_rb_tree() {
            return 5;
        }
    }

    for i in range(n) {
        tree:delete(i);
        if !tree:is_valid_rb_tree() {
            return 6;
        }
    }

    if tree.root:has_value() {
        return 7;
    }

    return 0;
}

/// @COMPILE
/// @RUN
