typedef[T] Ptr : {data: T}

typedef[T] LinkedListNode : {
    next: Ptr<LinkedListNode<T>>,
    data: T
}

function main : () -> int = {
    let a : LinkedListNode<int>;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'typedef'
/// Error: cannot construct recursive type 'typedef LinkedListNode<int> : {next: Ptr<LinkedListNode<int>>, data: int}' since it has a non-determinable size
