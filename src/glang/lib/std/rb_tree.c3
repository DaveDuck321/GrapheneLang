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

function left_if : (condition : bool) -> bool = {
    if condition {
        return LEFT();
    }

    return RIGHT();
}

typedef [K] RB_Node : {
    key    : K,
    left   : Optional<RB_Node<K>&>,
    right  : Optional<RB_Node<K>&>,
    parent : Optional<RB_Node<K>&>,
    colour : bool
}

typedef [K] RB_Tree : {
    root : Optional<RB_Node<K>&>,
    allocator : Allocator,  // FIXME use global allocator.
}

@operator [K] == : (left : RB_Node<K>&, right : RB_Node<K>&) -> bool = {
    return ref_to_addr(&left) == ref_to_addr(&right);
}

function [IO, K] write : (
    io : IO&, node : RB_Node<K>
) -> TypeIf<StringView, IsStreamableTextIO<IO>> = {
    // NOTE for debugging only.
    io:write(node.key);
    if node.colour == RED() {
        io:write(sv(" (red)"));
    }
    else {
        io:write(sv(" (black)"));
    }
    io:write(sv("\nleft: {\n"));
    io:write(node.left);
    io:write(sv("\n}\nright: {\n"));
    io:write(node.right);
    return io:write(sv("\n}\n"));
}

function [K] new<RB_Node<K>> : (
    self : RB_Tree<K>&, key : K, parent : Optional<RB_Node<K>&>
) -> RB_Node<K>& = {
    let node : RB_Node<K>& = &self.allocator:allocate<RB_Node<K>>();

    node = {
        .key = key,
        .left = make_optional<RB_Node<K>&>(),
        .right = make_optional<RB_Node<K>&>(),
        .parent = parent,
        .colour = RED(), // NOTE new nodes are red.
    };

    return &node;
}

function [K] make<RB_Tree<K>> : () -> RB_Tree<K> = {
    return {
        .root = make_optional<RB_Node<K>&>(),
        .allocator = initialize_allocator(),
    };
}

function [K] get_child : (
    node : RB_Node<K>&, side : bool
) -> Optional<RB_Node<K>&>& = {
    if side == LEFT() {
        return &node.left;
    }

    return &node.right;
}

function [K] which_child_am_i : (node : RB_Node<K>&) -> bool = {
    runtime_assert(node.parent:has_value());
    let parent : RB_Node<K>& = &node.parent:data();

    // NOTE we have to borrow explicitly to call operator == with references.
    if parent.left:has_value() and &parent.left:data() == &node {
        return LEFT();
    }

    runtime_assert(parent.right:has_value() and &parent.right:data() == &node);
    return RIGHT();
}

function [K] is_red : (node_opt : Optional<RB_Node<K>&>&) -> bool = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    return node_opt:has_value() and node_opt:data().colour == RED();
}

function [K] is_black : (node_opt : Optional<RB_Node<K>&>&) -> bool = {
    return !node_opt:is_red();
}

function [K] colour_black : (node_opt : Optional<RB_Node<K>&>&) -> void = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    if node_opt:has_value() {
        node_opt:data().colour = BLACK();
    }
}

function [K] rotate : (
    self : RB_Tree<K>&, u : RB_Node<K>&, side : bool
) -> void = {
    // If we are rotating left, get the right child.
    let v : RB_Node<K>& = &u:get_child(!side):data();

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

function [K] fix_after_insert : (
    self : RB_Tree<K>&, node : RB_Node<K>&
) -> void = {
    // We need to use a Ptr here so that we can update the value on every
    // iteration.
    let u_ptr : Ptr<RB_Node<K>> = {&node};

    while u_ptr:data().colour == RED() {
        let u : RB_Node<K>& = &u_ptr:data();

        // Case 4: u has no parent.
        if !u.parent:has_value() {
            // u is the root node. Colour it black.
            u.colour = BLACK();
            return;
        }

        let v : RB_Node<K>& = &u.parent:data();

        // Case 1: u has a black parent.
        if v.colour == BLACK() {
            // u is red, so both of its children must be black. Given that its
            // parent is also black, we can just leave u red.
            return;
        }

        // u has a red parent. Property 2 of red-black trees states that the
        // root node is black, therefore v also has a parent.
        let w : RB_Node<K>& = &v.parent:data();

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

function [K] insert : (self : RB_Tree<K>&, key : K) -> void = {
    let p_opt : Optional<RB_Node<K>&> = make_optional<RB_Node<K>&>();
    let u_ptr : Ptr<Optional<RB_Node<K>&>> = {&self.root};

    while u_ptr:data():has_value() {
        let u : RB_Node<K>& = &u_ptr:data():data();

        if key == u.key {
            // Key already exists. Overwrite it with the new key (if the tree is
            // used to implement a map, then the associated value could be
            // different) and return early, as the red-black tree is already
            // valid.
            u.key = key;
            return;
        }

        // Keep track of the parent node.
        p_opt:store(&u_ptr:data():data());

        u_ptr = {&u:get_child(left_if(key < u.key))};
    }

    u_ptr:data():store(&self:new<RB_Node<K>>(key, p_opt));

    if p_opt:has_value() {
        let p : RB_Node<K>& = &p_opt:data();
        p:get_child(left_if(key < p.key)) = u_ptr:data();
    }

    self:fix_after_insert(&u_ptr:data():data());
}

function [K] search : (self : RB_Tree<K>&, key : K) -> Optional<K> = {
    let u : Optional<RB_Node<K>&> = self.root;

    while u:has_value() {
        let u_key : K& = &u:data().key;

        if key == u_key {
            return make_optional(u_key);
        }

        u = u:data():get_child(left_if(key < u_key));
    }

    return make_optional<K>();
}
