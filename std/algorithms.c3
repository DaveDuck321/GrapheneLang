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

function [T] index_of : (span : Span<T>&, thing : T) -> isize = {
    // TODO: support enumerate
    for i in range(span.length) {
        if span.data[i] == thing {
            return i;
        }
    }
    return -1;
}

function [T] reverse : (buffer : Span<T>&) -> void = {
    let rhs : isize = buffer.length - 1;
    let lhs : isize = 0;
    while rhs > lhs {
        let old_lhs : T = buffer.data[lhs];
        buffer.data[lhs] = buffer.data[rhs];
        buffer.data[rhs] = old_lhs;

        rhs -= 1;
        lhs += 1;
    }
}

function [T] bisect_left : (span : Span<T>, elem : T) -> isize = {
    let lo : isize = 0;
    let hi : isize = span.length;

    while lo < hi {
        // Avoid overflows when averaging.
        const mid : isize = lo + (hi - lo) / 2;

        if span.data[mid] < elem {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}
