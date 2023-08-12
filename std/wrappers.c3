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

// BUG: the error from `function [T, @Len] make_span : (data : T[@Len]&) -> Span` gives the wrong line number
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
