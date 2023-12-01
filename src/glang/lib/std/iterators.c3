@require_once "arithmetic.c3"
@require_once "assert.c3"
@require_once "type_traits.c3"

typedef [C, R] ContiguousContainerIter : {
    container : C,
    current_index : isize,
}
typedef [T] Range : {lower : TypeIf<T, IsIntegral<T>>, upper : T, step : T}

// Arrays
function [T, @Len] length : (array : T[@Len]&) -> isize = {
    return @Len;
}

function [T, @Len] length : (array : T[@Len] mut&) -> isize = {
    return @Len;
}

function [T, @Len] get : (array : T[@Len]&, index : isize) -> T& = {
    runtime_assert(index < @Len);
    return &array[index];
}

function [T, @Len] get : (array : T[@Len] mut&, index : isize) -> T mut& = {
    runtime_assert(index < @Len);
    return &mut array[index];
}

function [T, @Len] get_iter : (array : T[@Len]&) -> ContiguousContainerIter<T[@Len]&, T&> = {
    return {
        .container = &array,
        .current_index = 0,
    };
}

function [T, @Len] get_iter : (array : T[@Len] mut&) -> ContiguousContainerIter<T[@Len] mut&, T mut&> = {
    return {
        .container = &mut array,
        .current_index = 0,
    };
}

// Range
function [T] range : (upper : T) -> Range<T> = {
    return {0, upper, 1};
}

function [T] range : (lower : T, upper : T) -> Range<T> = {
    return {lower, upper, 1};
}

function [T] range : (lower : T, upper : T, step : T) -> Range<T> = {
    return {lower, upper, step};
}

@implicit [T] get_next : (range: Range<T> mut&) -> T = {
    let next : T = range.lower;
    range.lower += range.step;
    return next;
}

@implicit [T] has_next : (range : Range<T> mut&) -> bool = {
    if range.step > 0 {
        return range.lower < range.upper;
    } else {
        return range.lower > range.upper;
    }
}

// ContiguousContainerIter
// TODO: NO ITERATOR FUNCTIONS ARE CONSTANT!!
//   This hurts the const-correctness of functions taking iterators directly
@implicit [T, R] has_next : (iter : ContiguousContainerIter<T, R> mut&) -> bool = {
    return iter.current_index < iter.container:length();
}

@implicit [T, R] get_next : (iter : ContiguousContainerIter<T, R> mut&) -> TypeIf<R, IsMutRef<R>> = {
    let index : isize = iter.current_index;
    iter.current_index += 1;
    return &mut iter.container:get(index);
}

@implicit [T, R] get_next : (iter : ContiguousContainerIter<T, R> mut&) -> TypeIf<R, IsConstRef<R>> = {
    let index : isize = iter.current_index;
    iter.current_index += 1;
    return &iter.container:get(index);
}

function [T, R] peak : (iter : ContiguousContainerIter<T, R> mut&) -> TypeIf<R, IsMutRef<R>> = {
    runtime_assert(iter:has_next());
    return &mut iter.container:get(iter.current_index);
}

function [T, R] peak : (iter : ContiguousContainerIter<T, R> mut&) -> TypeIf<R, IsConstRef<R>> = {
    runtime_assert(iter:has_next());
    return &iter.container:get(iter.current_index);
}
