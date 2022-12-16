// Error: generic 'T' has a generic annotation (generic types may not have
// additional generic annotations)
typedef struct_t[T, U]: {a: T[U]};

function main: () -> int = {
    let s: struct_t[int, int]; // unused typedef's are not parsed

    return 0;
};
