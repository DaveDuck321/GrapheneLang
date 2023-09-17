typedef[T] Ptr : {data: T&}
typedef[T] LinkedListNode1 : {
    next: Ptr<LinkedListNode1<T>>,
    data: T
}

typedef[T] Wrapper : {data: T}
typedef[T] LinkedListNode2 : {
    next: Wrapper<LinkedListNode2<T>>&,
    data: T
}

typedef[T] LinkedListNode3 : {
    next: Wrapper<LinkedListNode3<T>&>,
    data: T
}

function main : () -> int = {
    mut a : LinkedListNode1<int>;
    mut b : LinkedListNode2<int>;
    mut c : LinkedListNode3<int>;
    return 0;
}

/// @COMPILE
/// @RUN
