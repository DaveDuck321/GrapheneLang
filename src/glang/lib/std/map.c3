@require_once "arithmetic.c3"
@require_once "iterators.c3"
@require_once "rb_tree.c3"
@require_once "type_traits.c3"
@require_once "wrappers.c3"

typedef [K, V] MapNode : {
    key   : K,
    value : V
}

typedef [K, V] MapIter : RB_Iter<MapNode<K, V>>

// FIXME <=> is untested and broken :(
typedef [K, V] HasSpaceship<MapNode<K, V>> : HasSpaceship<K>

@operator [K1, K2, V] == : (left : MapNode<K1, V>&, right : MapNode<K2, V>&) -> bool = {
    return &left.key == &right.key;
}

@operator [K1, K2, V] != : (left : MapNode<K1, V>&, right : MapNode<K2, V>&) -> bool = {
    return &left.key != &right.key;
}

@operator [K1, K2, V] < : (left : MapNode<K1, V>&, right : MapNode<K2, V>&) -> bool = {
    return &left.key < &right.key;
}

@operator [K, V] != : (left : K&, right : MapNode<K, V>&) -> bool = {
    return &left != &right.key;
}

@operator [K, V] < : (left : K&, right : MapNode<K, V>&) -> bool = {
    return &left < &right.key;
}

typedef [K, V] Map : {
    tree   : RB_Tree<MapNode<K, V>>,
    length : isize
}

function [K, V] make<Map<K, V>> : () -> Map<K, V> = {
    return {
        .tree = make<RB_Tree<MapNode<K, V>>>(),
        .length = 0,
    };
}

function [K, V, C] at : (self : Map<K, V>&, key : C) -> Optional<V> = {
    let result : Optional<MapNode<K, V>> = self.tree:search(key);

    if result:has_value() {
        return make<Optional<V>>(result:data().value);
    }

    return make<Optional<V>>();
}

function [K, V] empty : (self : Map<K, V>&) -> bool = {
    return self.length == 0;
}

function [K, V] size : (self : Map<K, V>&) -> isize = {
    return self.length;
}

function [K, V] clear : (self : Map<K, V> mut&) -> void = {
    mut min : Optional<MapNode<K, V>> = self.tree:minimum();
    while min:has_value() {
        self:erase(min:data());
        min = self.tree:minimum();
    }
}

function [K, V] insert : (self : Map<K, V> mut&, key : K, value : V) -> void = {
    let node : MapNode<K, V> = {
        .key = key,
        .value = value
    };

    self.tree:insert(node);
    self.length += 1;
}

function [K, V, C] erase : (self : Map<K, V> mut&, key : C) -> bool = {
    let deleted : bool = self.tree:delete(key);

    if deleted {
        self.length -= 1;
    }

    return deleted;
}

function [K, V, C] contains : (self : Map<K, V>&, key : C) -> bool = {
    let result : Optional<V> = self:at(key);
    return result:has_value();
}

function [K, V] get_citer : (map : Map<K, V> mut&) -> MapIter<K, V> = {
    return map.tree:get_citer();
}
