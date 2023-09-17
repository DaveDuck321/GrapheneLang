@require_once "span.c3"
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

function [Iter, T] index_of : (view : Iter, thing : T) -> isize = {
    // TODO: support enumerate
    mut index : int = 0;
    for candidate_item in view {
        if candidate_item == thing {
            return index;
        }
        index += 1;
    }
    return -1;
}

function [T] reverse : (buffer : Span<T>&) -> void = {
    mut rhs : isize = buffer.length - 1;
    mut lhs : isize = 0;
    while rhs > lhs {
        let old_lhs : T = buffer.data[lhs];
        buffer.data[lhs] = buffer.data[rhs];
        buffer.data[rhs] = old_lhs;

        rhs -= 1;
        lhs += 1;
    }
}

function [T] bisect_left : (view : View<T>, elem : T) -> isize = {
    mut lo : isize = 0;
    mut hi : isize = view.length;

    while lo < hi {
        // Avoid overflows when averaging.
        let mid : isize = lo + (hi - lo) / 2;

        if view.data[mid] < elem {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }

    return lo;
}
