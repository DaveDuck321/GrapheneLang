@require_once "type_traits.c3"

// Automatic arithmetic assignments
@assignment [U, V] += : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs + rhs;
}

@assignment [U, V] -= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs - rhs;
}

@assignment [U, V] *= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs * rhs;
}

@assignment [U, V] /= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs / rhs;
}

@assignment [U, V] @= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs @ rhs;
}

@assignment [U, V] %= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs % rhs;
}


// Automatic logical assignments
@assignment [U, V] |= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs | rhs;
}

@assignment [U, V] <<= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs << rhs;
}

@assignment [U, V] >>= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs >> rhs;
}

@assignment [U, V] ^= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs ^ rhs;
}

@assignment [U, V] &= : (lhs : U&, rhs : V) -> TypeIf<void, HasTrivialAssignments<U>> = {
    lhs = lhs & rhs;
}
