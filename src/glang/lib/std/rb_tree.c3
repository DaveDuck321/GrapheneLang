@require_once "std/memory.c3"
@require_once "std/wrappers.c3"

function RED : () -> bool = {
    return false;
}

function BLACK : () -> bool = {
    return true;
}

function LEFT : () -> bool = {
    return false;
}

function RIGHT : () -> bool = {
    return true;
}

typedef [K, V] RB_Node : {
    key    : K,
    value  : V,
    left   : Optional<RB_Node<K, V>&>,
    right  : Optional<RB_Node<K, V>&>,
    parent : Optional<RB_Node<K, V>&>,
    colour : bool
}

typedef [K, V] RB_Tree : {
    root : Optional<RB_Node<K, V>&>,
    allocator : Allocator,  // FIXME use global allocator.
}

@operator [K, V] == : (left : RB_Node<K, V>&, right : RB_Node<K, V>&) -> bool = {
    // Assume both nodes belong to the same tree, to keep things simple.
    return left.key == right.key;
}

function [IO, K, V] write : (
    io : IO&, node : RB_Node<K, V>
) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    // NOTE for debugging only.
    io:write(node.key);
    if node.colour == RED() {
        io:write(sv(" (red)"));
    }
    else {
        io:write(sv(" (black)"));
    }
    io:write(sv(": "));
    io:write(node.value);
    io:write(sv("\nleft: {\n"));
    io:write(node.left);
    io:write(sv("\n}\nright: {\n"));
    io:write(node.right);
    return io:write(sv("\n}\n"));
}

function [K, V] new<RB_Node<K, V>> : (
    self : RB_Tree<K, V>&, key : K, value : V, parent : Optional<RB_Node<K, V>&>
) -> RB_Node<K, V>& = {
    let node : RB_Node<K, V>& = &self.allocator:allocate<RB_Node<K, V>>();

    node = {
        .key = key,
        .value = value,
        .left = make_optional<RB_Node<K, V>&>(),
        .right = make_optional<RB_Node<K, V>&>(),
        .parent = parent,
        .colour = RED(), // NOTE new nodes are red.
    };

    return &node;
}

function [K, V] make<RB_Tree<K, V>> : () -> RB_Tree<K, V> = {
    return {
        .root = make_optional<RB_Node<K, V>&>(),
        .allocator = initialize_allocator(),
    };
}

function [K, V] get_child : (
    node : RB_Node<K, V>&, side : bool
) -> Optional<RB_Node<K, V>&>& = {
    if side == LEFT() {
        return &node.left;
    }

    return &node.right;
}

function [K, V] which_child_am_i : (node : RB_Node<K, V>&) -> bool = {
    runtime_assert(node.parent:has_value());
    let parent : RB_Node<K, V>& = &node.parent:data();

    // NOTE we have to borrow explicitly to call operator == with references.
    if parent.left:has_value() and &parent.left:data() == &node {
        return LEFT();
    }

    runtime_assert(parent.right:has_value() and &parent.right:data() == &node);
    return RIGHT();
}

function [K, V] is_red : (node_opt : Optional<RB_Node<K, V>&>&) -> bool = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    return node_opt:has_value() and node_opt:data().colour == RED();
}

function [K, V] is_black : (node_opt : Optional<RB_Node<K, V>&>&) -> bool = {
    return !node_opt:is_red();
}

function [K, V] colour_black : (
    node_opt : Optional<RB_Node<K, V>&>&
) -> void = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    if node_opt:has_value() {
        node_opt:data().colour = BLACK();
    }
}

function [K, V] rotate : (
    self : RB_Tree<K, V>&, u : RB_Node<K, V>&, side : bool
) -> void = {
    // If we are rotating left, get the right child.
    let v : RB_Node<K, V>& = &get_child(&u, !side):data();

    // Move child from v to u, and update the child's parent pointer.
    u:get_child(!side) = v:get_child(side);
    if u:get_child(!side):has_value() {
        u:get_child(!side):data().parent:store(&u);
    }

    // Replace u with v.
    v.parent = u.parent;
    if v.parent:has_value() {
        const u_side : bool = which_child_am_i(&u);
        u.parent:data():get_child(u_side):store(&v);
    }
    else {
        self.root:store(&v);
    }

    // Make u a child of v;
    v:get_child(side):store(&u);
    u.parent:store(&v);
}

function [K, V] fix_after_insert : (
    self : RB_Tree<K, V>&, node : RB_Node<K, V>&
) -> void = {
    // We need to use a Ptr here so that we can update the value on every
    // iteration.
    let u_ptr : Ptr<RB_Node<K, V>> = {&node};

    while u_ptr:data().colour == RED() {
        let u : RB_Node<K, V>& = &u_ptr:data();

        // Case 4: u has no parent.
        if !u.parent:has_value() {
            // u is the root node. Colour it black.
            u.colour = BLACK();
            return;
        }

        let v : RB_Node<K, V>& = &u.parent:data();

        // Case 1: u has a black parent.
        if v.colour == BLACK() {
            // u is red, so both of its children must be black. Given that its
            // parent is also black, we can just leave u red.
            return;
        }

        // u has a red parent. Property 2 of red-black trees states that the
        // root node is black, therefore v also has a parent.
        let w : RB_Node<K, V>& = &v.parent:data();

        // Case 2: u has a red parent and a red uncle.
        if w.left:is_red() and w.right:is_red() {
            w.left:colour_black();
            w.right:colour_black();

            // Push the issue up the tree.
            w.colour = RED();
            u_ptr = {&w};
            // TODO continue;
        }
        else {
            // Case 3: u has a red parent and a black uncle.
            const u_side : bool = which_child_am_i(&u);
            const uncle_side : bool = !which_child_am_i(&v);

            if u_side != uncle_side {
                // Case 3a: u is away from its uncle.
                // Rotate, then leave u red and colour w, its new sibling, red.
                // Their new parent should thus be black.
                self:rotate(&w, !u_side);
                v.colour = BLACK();
                w.colour = RED();
            }
            else {
                // Case 3b: u is towards its uncle.
                // Now u becomes the parent of v and w. Colour it black, leave v
                // red, and make w red.
                self:rotate(&v, !u_side);
                self:rotate(&w, u_side);
                u.colour = BLACK();
                w.colour = RED();
            }

            return;
        }
    }
}

function [K, V] insert : (
    self : RB_Tree<K, V>&, key : K, value : V
) -> void = {
    let p_opt : Optional<RB_Node<K, V>&> = make_optional<RB_Node<K, V>&>();
    let u_ptr : Ptr<Optional<RB_Node<K, V>&>> = {&self.root};

    while u_ptr:data():has_value() {
        let u : RB_Node<K, V>& = &u_ptr:data():data();

        if key == u.key {
            // Key already exists. We can just replace the value and return
            // early, as the red-black tree is already valid.
            u.value = value;
            return;
        }
        else {
            // Keep track of the parent node.
            p_opt:store(&u_ptr:data():data());

            if key < u.key {
                u_ptr = {&u.left};
            }
            else {
                u_ptr = {&u.right};
            }
        }
    }

    u_ptr:data():store(&self:new<RB_Node<K, V>>(key, value, p_opt));

    if p_opt:has_value() {
        if key < p_opt:data().key {
            p_opt:data().left = u_ptr:data();
        }
        else {
            p_opt:data().right = u_ptr:data();
        }
    }

    self:fix_after_insert(&u_ptr:data():data());
}

function [K, V] search : (self : RB_Tree<K, V>&, key : K) -> Optional<V> = {
    let u : Optional<RB_Node<K, V>&> = self.root;

    while u:has_value() {
        let u_key : K& = &u:data().key;

        if key == u_key {
            return make_optional(u:data().value);
        }

        if key < u_key {
            u = u:data().left;
        }
        else {
            u = u:data().right;
        }
    }

    return make_optional<V>();
}
