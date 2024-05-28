@require_once "type_traits.c3"

typedef [T] Ptr : { data : T& }
typedef [T] MutPtr : { data : T mut& }
typedef [T] MaybeNullPtr : Optional<Ptr<T>>

// Maybe FIXME: this isn't really ergonomic, this SFINAE avoids &/ mut& code duplication
typedef [T] Optional : { _value : TypeIf<T, Not<IsRef<T>>>, _has_value : bool}
typedef [T] Optional<T&> : { _ptr : iptr } // Nullptr optimization
typedef [T] Optional<T mut&> : { _ptr : iptr } // Duplicated

// Ptr
function [T] data : (self : Ptr<T>&) -> T& = {
    return &self.data;
}

function [T] data : (self : Ptr<T> mut&) -> T& = {
    return &self.data;
}

// MutPtr
function [T] data : (self : MutPtr<T>&) -> T mut& = {
    return &mut self.data;
}

function [T] data : (self : MutPtr<T> mut&) -> T mut& = {
    return &mut self.data;
}

// Generic Optional
function [T] make<Optional<T>> : () -> Optional<T> = {
    mut result : Optional<T>;
    result._has_value = false;
    return result;
}

function [T] make<Optional<T>> : (value : T) -> Optional<T> = {
    return {
        ._value = value,
        ._has_value = true,
    };
}

function [T] store : (self : Optional<T> mut&, value : T) -> void = {
    self._value = value;
    self._has_value = true;
}

function [T] erase : (self : Optional<T> mut&) -> void = {
    // TODO: call data's destructor
    self._has_value = false;
}

function [T] data : (self : Optional<T>&) -> Ref<T> = {
    runtime_assert(self:has_value());
    return &self._value;
}

function [T] data : (self : Optional<T> mut&) -> MutRef<T> = {
    runtime_assert(self:has_value());
    return &mut self._value;
}

function [T] has_value : (self : Optional<T>&) -> bool = {
    return self._has_value;
}

function [IO, T] write : (
    io : IO mut&, optional : Optional<T>
) -> TypeIf<void, IsStreamableTextIO<IO>> = {
    if optional:has_value() {
        return io:write(optional:data());
    }

    io:write("<empty>");
}


// Optional<T&>
function [T] make<Optional<T&>> : () -> Optional<T&> = {
    return { 0 };
}

function [T] make<Optional<T&>> : (value : T&) -> Optional<T&> = {
    return { ref_to_addr(&value) };
}

function [T] store : (self : Optional<T&> mut&, value : T&) -> void = {
    self._ptr = ref_to_addr(&value);
}

function [T] erase : (self : Optional<T&> mut&) -> void = {
    self._ptr = 0;
}

function [T] data : (self : Optional<T&>&) -> T& = {
    runtime_assert(self:has_value());
    return &addr_to_ref<T>(self._ptr);
}

function [T] data : (self : Optional<T&> mut&) -> T& = {
    return &addr_to_ref<T>(self._ptr);
}


function [T] has_value : (self : Optional<T&>&) -> bool = {
    return self._ptr != 0;
}

function [T] has_value : (self : Optional<T&> mut&) -> bool = {
    return self._ptr != 0;
}


// HUGE DUPLICATION FOR MUTABLE REFERENCE
// Optional<T&>
function [T] make<Optional<T mut&>> : () -> Optional<T mut&> = {
    return { 0 };
}

function [T] make<Optional<T mut&>> : (value : T mut&) -> Optional<T mut&> = {
    return { ref_to_addr(&value) };
}

function [T] store : (self : Optional<T mut&> mut&, value : T mut&) -> void = {
    self._ptr = ref_to_addr(&value);
}

function [T] erase : (self : Optional<T mut&> mut&) -> void = {
    self._ptr = 0;
}

function [T] data : (self : Optional<T mut&>&) -> T mut& = {
    return &mut addr_to_mut_ref<T>(self._ptr);
}

function [T] data : (self : Optional<T mut&> mut&) -> T mut& = {
    return &mut addr_to_mut_ref<T>(self._ptr);
}

function [T] has_value : (self : Optional<T mut&>&) -> bool = {
    return self._ptr != 0;
}

function [T] has_value : (self : Optional<T mut&> mut&) -> bool = {
    return self._ptr != 0;
}
