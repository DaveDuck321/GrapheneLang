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
    left   : Optional<RB_Node<K> mut&>,
    right  : Optional<RB_Node<K> mut&>,
    parent : Optional<RB_Node<K> mut&>,
    colour : bool
}

typedef [K] RB_Tree : {
    root : Optional<RB_Node<K> mut&>,
    allocator : Allocator,  // FIXME use global allocator.
}

@operator [K] == : (left : RB_Node<K>&, right : RB_Node<K>&) -> bool = {
    return ref_to_addr(&left) == ref_to_addr(&right);
}

@operator [K] != : (left : RB_Node<K>&, right : RB_Node<K>&) -> bool = {
    return !(&left == &right);
}

@operator [K] == : (
    left : Optional<RB_Node<K> mut&>, right : Optional<RB_Node<K> mut&>
) -> bool = {
    if left:has_value() and right:has_value() {
        return &left:data() == &right:data();
    }

    return left:has_value() == right:has_value();
}

@operator [K] != : (
    left : Optional<RB_Node<K> mut&>, right : Optional<RB_Node<K> mut&>
) -> bool = {
    return !(left == right);
}

function [K] new<RB_Node<K>> : (
    self : RB_Tree<K> mut&, key : K, parent : Optional<RB_Node<K> mut&>
) -> RB_Node<K> mut& = {
    let node : RB_Node<K> mut& = &mut self.allocator:allocate<RB_Node<K>>();

    node = {
        .key = key,
        .left = make<Optional<RB_Node<K> mut&>>(),
        .right = make<Optional<RB_Node<K> mut&>>(),
        .parent = parent,
        .colour = RED(), // NOTE new nodes are red.
    };

    return &mut node;
}

function [K] make<RB_Tree<K>> : () -> RB_Tree<K> = {
    return {
        .root = make<Optional<RB_Node<K> mut&>>(),
        .allocator = initialize_allocator(),
    };
}

function [K] get_child : (
    node : RB_Node<K> mut&, side : bool
) -> Optional<RB_Node<K> mut&> mut& = {
    if side == LEFT() {
        return &mut node.left;
    }

    return &mut node.right;
}

function [K] which_child_am_i : (
    node : Optional<RB_Node<K> mut&>, parent : RB_Node<K> mut&
) -> bool = {
    // Use this overload if node could be a leaf node (nil node).

    // NOTE we have to borrow explicitly to call operator == with references.
    if parent.left == node {
        return LEFT();
    }

    runtime_assert(parent.right == node);
    return RIGHT();
}

function [K] which_child_am_i : (node : RB_Node<K> mut&) -> bool = {
    return which_child_am_i(make(&mut node), &mut node.parent:data());
}

function [K] is_red : (node_opt : Optional<RB_Node<K> mut&>&) -> bool = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    return node_opt:has_value() and node_opt:data().colour == RED();
}

function [K] is_black : (node_opt : Optional<RB_Node<K> mut&>&) -> bool = {
    return !node_opt:is_red();
}

function [K] colour_black : (node_opt : Optional<RB_Node<K> mut&> mut&) -> void = {
    // Using property 3 of red-black trees: every leaf node (nil node) is black.
    if node_opt:has_value() {
        node_opt:data().colour = BLACK();
    }
}

function [K] rotate : (
    self : RB_Tree<K> mut&, u : RB_Node<K> mut&, side : bool
) -> void = {
    // If we are rotating left, get the right child.
    let v : RB_Node<K> mut& = &mut u:get_child(!side):data();

    // Move child from v to u, and update the child's parent pointer.
    u:get_child(!side) = v:get_child(side);
    if u:get_child(!side):has_value() {
        u:get_child(!side):data().parent:store(&mut u);
    }

    // Replace u with v.
    v.parent = u.parent;
    if v.parent:has_value() {
        let u_side : bool = u:which_child_am_i();
        u.parent:data():get_child(u_side):store(&mut v);
    }
    else {
        self.root:store(&mut v);
    }

    // Make u a child of v;
    v:get_child(side):store(&mut u);
    u.parent:store(&mut v);
}

function [K] fix_after_insert : (
    self : RB_Tree<K> mut&, node : RB_Node<K> mut&
) -> void = {
    // We need to use a Ptr here so that we can update the value on every
    // iteration.
    mut u_ptr : MutPtr<RB_Node<K>> = {&mut node};

    while u_ptr:data().colour == RED() {
        let u : RB_Node<K> mut& = &mut u_ptr:data();

        // Case 4: u has no parent.
        if !u.parent:has_value() {
            // u is the root node. Colour it black.
            u.colour = BLACK();
            return;
        }

        let v : RB_Node<K> mut& = &mut u.parent:data();

        // Case 1: u has a black parent.
        if v.colour == BLACK() {
            // u is red, so both of its children must be black. Given that its
            // parent is also black, we can just leave u red.
            return;
        }

        // u has a red parent. Property 2 of red-black trees states that the
        // root node is black, therefore v also has a parent.
        let w : RB_Node<K> mut& = &mut v.parent:data();

        // Case 2: u has a red parent and a red uncle.
        if w.left:is_red() and w.right:is_red() {
            w.left:colour_black();
            w.right:colour_black();

            // Push the issue up the tree.
            w.colour = RED();
            u_ptr = {&mut w};
            continue;
        }

        // Case 3: u has a red parent and a black uncle.
        let u_side : bool = u:which_child_am_i();
        let uncle_side : bool = !v:which_child_am_i();

        if u_side != uncle_side {
            // Case 3a: u is away from its uncle.
            // Rotate, then leave u red and colour w, its new sibling, red.
            // Their new parent should thus be black.
            self:rotate(&mut w, !u_side);
            v.colour = BLACK();
            w.colour = RED();
        }
        else {
            // Case 3b: u is towards its uncle.
            // Now u becomes the parent of v and w. Colour it black, leave v
            // red, and make w red.
            self:rotate(&mut v, !u_side);
            self:rotate(&mut w, u_side);
            u.colour = BLACK();
            w.colour = RED();
        }

        return;
    }
}

function [K] insert : (self : RB_Tree<K> mut&, key : K) -> void = {
    mut p_opt : Optional<RB_Node<K> mut&> = make<Optional<RB_Node<K> mut&>>();
    mut u_ptr : MutPtr<Optional<RB_Node<K> mut&>> = {&mut self.root};

    while u_ptr:data():has_value() {
        let u : RB_Node<K> mut& = &mut u_ptr:data():data();

        if key == u.key {
            // Key already exists. Overwrite it with the new key (if the tree is
            // used to implement a map, then the associated value could be
            // different) and return early, as the red-black tree is already
            // valid.
            u.key = key;
            return;
        }

        // Keep track of the parent node.
        p_opt:store(&mut u_ptr:data():data());

        u_ptr = {&mut u:get_child(left_if(key < u.key))};
    }

    u_ptr:data():store(&mut self:new<RB_Node<K>>(key, p_opt));

    self:fix_after_insert(&mut u_ptr:data():data());
}

function [K, Comparable] search_impl : (
    self : RB_Tree<K> mut&, key : Comparable
) -> Optional<RB_Node<K> mut&> = {
    mut u : Optional<RB_Node<K> mut&> = self.root;

    while u:has_value() and key != u:data().key {
        u = u:data():get_child(left_if(key < u:data().key));
    }

    return u;
}

function [K, Comparable] search : (
    self : RB_Tree<K> mut&, key : Comparable
) -> Optional<K> = {
    mut result : Optional<RB_Node<K> mut&> = self:search_impl(key);

    if result:has_value() {
        return make(result:data().key);
    }

    return make<Optional<K>>();
}

function [K] replace_node_with : (
    self : RB_Tree<K> mut&, old : RB_Node<K> mut&, new : Optional<RB_Node<K> mut&>
) -> void = {
    // Make old's parent point to new instead of old.
    if old.parent:has_value() {
        old.parent:data():get_child(old:which_child_am_i()) = new;
    }
    else {
        self.root = new;
    }

    // Set new's new parent, if it's not a leaf node (nil node).
    if new:has_value() {
        new:data().parent = old.parent;
    }
}

function [K] minmax : (node : RB_Node<K> mut&, direction : bool) -> RB_Node<K> mut& = {
    mut u : MutPtr<RB_Node<K>> = {&mut node};

    // TODO some sort of lambda/local function would really help here.
    while u:data():get_child(direction):has_value() {
        u = {&mut u:data():get_child(direction):data()};
    }

    return &mut u:data();
}

function [K] minmax : (tree : RB_Tree<K> mut&, direction : bool) -> Optional<K> = {
    if tree.root:has_value() {
        return make(tree.root:data():minmax(direction).key);
    }
    return make<Optional<K>>();
}

function [K] minimum : (tree : RB_Tree<K> mut&) -> Optional<K> = {
    return tree:minmax(LEFT());
}

function [K] maximum : (tree : RB_Tree<K> mut&) -> Optional<K> = {
    return tree:minmax(RIGHT());
}

function [K] fix_after_delete_impl : (
    self : RB_Tree<K> mut&, x : Optional<RB_Node<K> mut&>, x_parent : RB_Node<K> mut&
) -> Optional<RB_Node<K> mut&> = {
    // x is guaranteed not to be the root.
    mut x_side : bool = which_child_am_i(x, &mut x_parent);
    mut w_side : bool = !x_side;

    // x's sibling, w, cannot be a leaf node (nil node). If x is double black,
    // then its sibling tree must also have a height of at least 2 black nodes.
    let w : RB_Node<K> mut& = &mut x_parent:get_child(w_side):data();

    // Case 1: w is red.
    if w.colour == RED() {
        // The rotation makes x's new sibling black.
        self:rotate(&mut x_parent, x_side);
        x_parent.colour = RED();
        w.colour = BLACK();

        // x is unchanged; we loop again and enter one of the other three cases.
        return x;
    }

    // Case 2: w is black, w's children are black.
    if w.left:is_black() and w.right:is_black() {
        // Colour w red, to decrement the height of the sibling tree (in term of
        // black nodes). This allows us to move x up to our parent: if x_parent
        // was black, then now it is double black; if it was red, then now it is
        // red-and-black.
        w.colour = RED();
        return make(&mut x_parent);
    }

    // Case 3: w is black with red close and black distant child.
    if w:get_child(x_side):is_red() and w:get_child(w_side):is_black() {
        // Make w's red child w's new parent, and colour it black, such that x's
        // sibling is now black. w must in turn become red, to compensate.
        self:rotate(&mut w, w_side);
        w.colour = RED();
        w.parent:colour_black();

        // x remains in place, and we get case 4.
        return x;
    }

    // Case 4: w is black and w's distant child is red.
    runtime_assert(w:get_child(w_side):is_red());

    // Essentially, we make w x_parent's parent, which means that there is an
    // additional black node now in the path of x. This allows us to remove the
    // double black. The root of the sibling tree is now w's red child, which
    // we colour black to compensate.
    self:rotate(&mut x_parent, x_side);

    // w takes x_parent's colour, so the colour of this subtree's root is
    // preserved.
    w.colour = x_parent.colour;

    // We have removed one black node from w's side of the tree (w itself), so
    // we need to make its red child black to compensate (as it is now the root
    // of that subtree). On the other side, the new root should also be black,
    // as it absorbs the "double black" from x.
    w.left:colour_black();
    w.right:colour_black();

    // Remove the double black; we are done.
    return self.root;
}

function [K] fix_after_delete : (
    self : RB_Tree<K> mut&,
    x : Optional<RB_Node<K> mut&>,
    x_parent : Optional<RB_Node<K> mut&>
) -> void = {
    // x is "double black" inside the loop.
    while x != self.root and x:is_black() {
        // x might be a leaf node (nil node), so take its parent as an argument.
        // x_parent is an optional, as the root node has no parent. However, it
        // should have a value in all other cases.
        x = self:fix_after_delete_impl(x, &mut x_parent:data());

        // In cases 1 and 3, x is unchanged, so it could still be an empty
        // optional, representing a leaf node (nil node). In the other cases, x
        // moves up the tree, so we must update x_parent.
        if x:has_value() {
            x_parent = x:data().parent;
        }
    }

    // If x was the root, then it must always be black. If x was
    // "red-and-black" (i.e. x was originally red), we can colour it black to
    // resolve the issue.
    x:colour_black();
}

function [K] delete_impl : (self : RB_Tree<K> mut&, u : RB_Node<K> mut&) -> void = {
    // x is the "double black" or "red-and-black" node which we use to fix any
    // red-black tree violations that arise from the deletion. x_parent is used
    // to keep track of its parent node, as x might be a leaf node (nil node),
    // which is represented as an empty optional.
    mut x : Optional<RB_Node<K> mut&> = make<Optional<RB_Node<K> mut&>>();
    mut x_parent : Optional<RB_Node<K> mut&> = make<Optional<RB_Node<K> mut&>>();
    mut must_fix : bool = (u.colour == BLACK());

    if u.left:has_value() and u.right:has_value() {
        // u has two children.
        let v : RB_Node<K> mut& = &mut u.right:data():minmax(LEFT());
        must_fix = (v.colour == BLACK());
        x = v.right;

        if &v != &u.right:data() {
            // v doesn't have a left child, otherwise that child would be the
            // minimum. It might have a right child though, so replace v with
            // its right child before we move it around the tree. Set x_parent
            // before the shift, as the right child might be an empty optional.
            x_parent = v.parent;
            self:replace_node_with(&mut v, v.right);

            // v now adopts the branch on the right of u. We could have made v
            // the right child of u now, but there's no point, as we are about
            // to remove u and relink its children anyway. v is effectively the
            // right child of u now though.
            v.right = u.right;
            let x : RB_Node<K> mut& = &mut v.right:data();
            x.parent:store(&mut v);
        }
        else {
            x_parent = make(&mut v);
        }

        // Replace u with v.
        self:replace_node_with(&mut u, make(&mut v));
        v.left = u.left;
        v.left:data().parent:store(&mut v);
        v.colour = u.colour;
    }
    else {
        // u has zero or one children.
        // Get the child that might have the value.
        let side : bool = left_if(u.left:has_value());
        x = u:get_child(side);
        x_parent = u.parent;

        // And replace u with it. It's fine if x is empty; the code is just
        // simpler this way.
        self:replace_node_with(&mut u, x);
    }

    self.allocator:deallocate(&u);

    if must_fix {
        self:fix_after_delete(x, x_parent);
    }
}

function [K] delete : (self : RB_Tree<K> mut&, key : K) -> bool = {
    mut node : Optional<RB_Node<K> mut&> = self:search_impl(key);

    // Do nothing if the key is not in the tree.
    if node:has_value() {
        self:delete_impl(&mut node:data());
        return true;
    }

    return false;
}
