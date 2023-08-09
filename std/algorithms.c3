@require_once "wrappers.c3"

function [T] reverse : (buffer : Span<T>&) -> void = {
    let rhs : isize = buffer.length - 1;
    let lhs : isize = 0;
    while rhs > lhs {
        let old_lhs : T = buffer.data[lhs];
        buffer.data[lhs] = buffer.data[rhs];

        rhs -= 1;
        lhs += 1;
    }
}
