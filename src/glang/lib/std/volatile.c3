@require_once "std/util.c3"

function [T] volatile_read : (address: T&) -> T = {
    return __builtin_volatile_read(&address);
}

function [T] volatile_write : (address: T mut&, to_write_data: T) -> void = {
    __builtin_volatile_write(&mut address, to_write_data);
}


typedef [T] Volatile : {
    underlying: T mut&
}

function [T] read : (var: Volatile<T>&) -> T = {
    return volatile_read(&var.underlying);
}

function [T] write : (var: Volatile<T>&, to_write_data: T) -> void = {
    volatile_write(&mut var.underlying, to_write_data);
}

function [T] make<Volatile<T>> : (data: T mut&) -> Volatile<T> = {
    return {&mut data};
}

function [T] make<Volatile<T>> : (addr: iptr) -> Volatile<T> = {
    return {&mut addr_to_mut_ref<T>(addr)};
}

typedef [T] WriteVolatile : {
    underlying: T mut&
}

function [T] read : (var: WriteVolatile<T>&) -> T = {
    return var.underlying;
}

function [T] write : (var: WriteVolatile<T>&, to_write_data: T) -> void = {
    volatile_write(&mut var.underlying, to_write_data);
}

function [T] make<WriteVolatile<T>> : (data: T mut&) -> WriteVolatile<T> = {
    return {&mut data};
}

function [T] make<WriteVolatile<T>> : (addr: iptr) -> WriteVolatile<T> = {
    return {&mut addr_to_mut_ref<T>(addr)};
}


typedef [T] ReadOnlyVolatile : {
    underlying: T&
}

function [T] read : (var: ReadOnlyVolatile<T>&) -> T = {
    return var.underlying;
}

function [T] make<ReadOnlyVolatile<T>> : (data: T&) -> ReadOnlyVolatile<T> = {
    return {&data};
}

function [T] make<ReadOnlyVolatile<T>> : (addr: iptr) -> ReadOnlyVolatile<T> = {
    return {&addr_to_ref<T>(addr)};
}
