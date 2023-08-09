@require_once "wrappers.c3"

function [This, Next] is_one_of : (thing : This, next : Next) -> bool = {
    if thing == next {
        return true;
    }
    return false;
}

function [This, Next, Others...] is_one_of : (thing : This, next : Next, others : Others...) -> bool = {
    if thing == next {
        return true;
    }
    return is_one_of(thing, others...);
}

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
