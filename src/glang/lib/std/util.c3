@require_once "type_traits.c3"

function [T] sizeof : () -> isize = {
    return __builtin_sizeof<T>();
}

function [T] alignof : () -> isize = {
    return __builtin_alignof<T>();
}

function [T] have_same_addr : (a : T, b : T) -> TypeIf<bool, IsRef<T>> = {
    return ref_to_addr(&a) == ref_to_addr(&b);
}

function [T] ref_to_addr : (ref : T&) -> iptr = {
    return __builtin_ptr_to_int(&ref);
}

function [T] ref_to_addr : (ref : T[&]) -> iptr = {
    return __builtin_ptr_to_int(&ref);
}

// Note: this SFINAE disables automatic int promotion
function [T, AddrT] addr_to_ref : (address : AddrT)
    -> TypeIf<T&, AreEquivalent<AddrT, iptr>> =
{
    return &__builtin_int_to_ptr<T&>(address);
}

function [T, AddrT] addr_to_mut_ref : (address : AddrT)
    -> TypeIf<T mut&, AreEquivalent<AddrT, iptr>> =
{
    return &mut __builtin_int_to_ptr<T mut&>(address);
}

function [T, AddrT] addr_to_heap_array : (address : AddrT)
    -> TypeIf<T[&], AreEquivalent<AddrT, iptr>> =
{
    return &__builtin_int_to_ptr<T[mut&]>(address);
}

function [T, AddrT] addr_to_mut_heap_array : (address : AddrT)
    -> TypeIf<T[mut&], AreEquivalent<AddrT, iptr>> =
{
    return &mut __builtin_int_to_ptr<T[mut&]>(address);
}

function [TRef, BRef] reinterpret_addr : (base : BRef)
    -> TypeIf<TRef, Both<IsRef<TRef>, IsRef<BRef>>> =
{
    return &__builtin_int_to_ptr<TRef>(__builtin_ptr_to_int(&base));
}

function as_arithmetic : (value : u8)   -> i8   = { return __builtin_bitcast<i8>(value); }
function as_arithmetic : (value : u16)  -> i16  = { return __builtin_bitcast<i16>(value); }
function as_arithmetic : (value : u32)  -> i32  = { return __builtin_bitcast<i32>(value); }
function as_arithmetic : (value : u64)  -> i64  = { return __builtin_bitcast<i64>(value); }
function as_arithmetic : (value : u128) -> i128 = { return __builtin_bitcast<i128>(value); }

function as_logical : (value : i8)   -> u8   = { return __builtin_bitcast<u8>(value); }
function as_logical : (value : i16)  -> u16  = { return __builtin_bitcast<u16>(value); }
function as_logical : (value : i32)  -> u32  = { return __builtin_bitcast<u32>(value); }
function as_logical : (value : i64)  -> u64  = { return __builtin_bitcast<u64>(value); }
function as_logical : (value : i128) -> u128 = { return __builtin_bitcast<u128>(value); }

function [Ret, In] Narrow : (value : In) ->
TypeIf<
    Ret,
    Either<
        Both<IsIntegral<Ret>, IsIntegral<In>>,
        Both<IsLogical<Ret>, IsLogical<In>>
    >> =
{
    // TODO: reject sizeof(Ret) >= sizeof(In) at compile time
    return __builtin_narrow<Ret>(value);
}

function value<TrueType> : () -> bool = { return true; }
function value<FalseType> : () -> bool = { return false; }

// TODO: constant version of this is impossible atm (since __builtin_has_next is mutable)
function [T] has_next : (iter : T mut&) -> bool = { return iter:__builtin_has_next(); }
