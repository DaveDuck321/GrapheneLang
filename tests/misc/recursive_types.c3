typedef[T] Ptr : {data: T&}
typedef[T] LinkedListNode1 : {
    next: Ptr<LinkedListNode1<T>>,
    data: T
}

function main : () -> int = {
    let a : LinkedListNode1<int>;
    return 0;
}

/// @COMPILE
/// @RUN
