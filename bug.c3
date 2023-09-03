function foo : () -> bool = { return true; }

function bar : () -> bool = {
    // BUG does this even parse correctly?
    // YES
    // We should highlight && and || as invalid, and highlight `and` and `or` as
    // keywords. Then we can avoid any more errors like this.
    return true && foo();
}

function main : () -> int = { return 0; }
