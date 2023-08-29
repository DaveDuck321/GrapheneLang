typedef [T] Ptr : { _data : T& }

function [T] data : (self : Ptr<T>&) -> T& = {
    return &self._data;
}

typedef [T] Optional : { _value : T, _has_value : bool}

function [T] store : (self : Optional<T>&, value : T) -> void = {
    self._value = value;
    self._has_value = true;
}

function [T] erase : (self : Optional<T>&) -> void = {
    self._has_value = false;
}

// TODO: this should be a constructor
function [T] make_optional : () -> Optional<T> = {
    let result : Optional<T>;
    result._has_value = false;
    return result;
}

function [T] make_optional : (value : T) -> Optional<T> = {
    return {._value = value, ._has_value = true};
}

function [T] data : (self : Optional<T>&) -> T& = {
    return &self._value;
}

function [T] has_value : (self : Optional<T>&) -> bool = {
    return self._has_value;
}

typedef [T] Span : {
    data : T[&],
    length : isize,
}

typedef [T] SpanIter : {
    span : Span<T>,
    index : isize,
}

function [T] slice_from_offset : (span : Span<T>&, start_offset : isize) -> Span<T> = {
    // TODO: this should probably be a language feature
    let start_addr : iptr = ref_to_addr(&span.data[start_offset]);
    let new_length : isize = span.length - start_offset;

    // TODO: bounds check
    return {&addr_to_heap_array<T>(start_addr), new_length};
}

function [T] slice_between : (span : Span<T>&, start_offset : iptr, end_offset : isize) -> Span<T> = {
    // TODO: this should probably be a language feature
    let start_addr : iptr = ref_to_addr(&span.data[start_offset]);
    if end_offset > span.length {
        end_offset = span.length;
    }
    return {&addr_to_heap_array<T>(start_addr), end_offset - start_offset};
}

function [T, @Len] make_span : (data : T[@Len]&) -> Span<T> = {
    return {&data, @Len};
}

function [T] get_iter : (span : Span<T>&) -> SpanIter<T> = {
    return {span, 0};
}

@implicit [T] has_next : (iter : SpanIter<T>&) -> bool = {
    return iter.index < iter.span.length;
}

@implicit [T] get_next : (iter : SpanIter<T>&) -> T& = {
    let index : isize = iter.index;
    iter.index += 1;
    return &iter.span.data[index];
}
