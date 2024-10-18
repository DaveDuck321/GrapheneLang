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

typedef [T] ReadVolatile : {
    underlying: T mut&
}

function [T] read : (var: ReadVolatile<T>&) -> T = {
    return volatile_read(&var.underlying);
}

function [T] write : (var: ReadVolatile<T>&, to_write_data: T) -> void = {
    var.underlying = to_write_data;
}

function [T] make<ReadVolatile<T>> : (data: T mut&) -> ReadVolatile<T> = {
    return {&mut data};
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

typedef [T] NonVolatile : {
    underlying: T mut&
}

function [T] read : (var: NonVolatile<T>&) -> T = {
    return var.underlying;
}

function [T] write : (var: NonVolatile<T>&, to_write_data: T) -> void = {
    var.underlying = to_write_data;
}

function [T] make<NonVolatile<T>> : (data: T mut&) -> NonVolatile<T> = {
    return {&mut data};
}

function [T] volatile_read : (address: T&) -> T = {
    return __builtin_volatile_read(&address);
}

function [T] volatile_write : (address: T mut&, to_write_data: T) -> void = {
    __builtin_volatile_write(&mut address, to_write_data);
}
