typedef [T] Ptr : { _data : T& };

function [T] data : (self : Ptr<T>&) -> T& = {
    return &self._data;
};

typedef [T] Optional : { _value : T, _has_value : bool};

function [T] store : (self : Optional<T>&, value : T) -> void = {
    self._value = value;
    self._has_value = true;
};

function [T] erase : (self : Optional<T>&) -> void = {
    self._has_value = false;
};

// TODO: this should be a constructor
function [T] make_optional : () -> Optional<T> = {
    let result : Optional<T>;
    result._has_value = false;
    return result;
};

function [T] make_optional : (value : T) -> Optional<T> = {
    return {_value : value, _has_value : true};
};

function [T] data : (self : Optional<T>&) -> T& = {
    return &self._value;
};

function [T] has_value : (self : Optional<T>&) -> bool = {
    return self._has_value;
};

typedef [T] Span : {
    data : T[&],
    length : isize,
};
