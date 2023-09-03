typedef [T] Vector : { data : T[&] }

typedef Node : {
    self : Vector<Node>
}

function main : () -> int = {
    let node : Node;
    return 0;
}
