# Types

## Aliased Type

The syntax for aliasing types is:

```c
typedef Name1 : TypeThatWeWantToAlias;
typedef Name2 : TypeThatWeWantToAlias;
```
`Name1`, `Name2`, and `TypeThatWeWantToAlias` are equivalent and can be used interchangeably in any context.

### Generic aliases

The syntax for adding a generic parameter is:
```c
typedef [T] Generic : TypeThatWeWantToAlias;
```
- The specialized type `Generic<GenericArgumentType>`, is equivalent to `TypeThatWeWantToAlias`, `Name1`, and `Name2`.
- If the specialization is successful, the type `Generic<OtherType>` is indistinguishable from a non-generic typedef with the specialization `OtherType` manually substituted into the aliased type.

The syntax for overriding the alias for a specific specialization is:
```c
typedef Generic<SpecializationType> : OtherType;
```

- When specialized with a type equivalent to `SpecializationType`, `Generic<SpecializationType>` is equivalent to `OtherType`.
- A specifically specialized alias is used in preference to the generic alias when both are available.

## Reference Type

The syntax for creating a new reference type is:
```
DereferencedType&
```
Two reference types are equivalent as long as each `DereferencedType` is also equivalent


## Array Type
The syntax for creating a new array type is:
```
MemberType[(Dimension 0 length or & for an unknown size), (Dimension 1 length), ... (Dimension N length)]
```

- Two array types are equivalent if all of their dimensions are the same and each `MemberType` is also equivalent.

- We can implicitly convert from an array reference of type `T1&` to an array reference of type `T2&` if:
  - The `MemberType` of `T1` is equivalent to the `MemberType` of `T2` and
  - All dimensions of `T1`, except the first, are equal to the corresponding dimensions of `T2` and:
    - The first dimension of `T2` is smaller than the first dimension of `T1` or:
    - The first dimension of `T2` is an unknown size.

## Struct Types

The syntax for creating a new struct type is:
```
{member_name: MemberType, ...}
```
- Two struct types are NEVER equivalent
- Structs support NO implicit conversions


# Using Types

## Initializer Lists

An initializer list has the following syntax:

```
<as an expression>
{member1: value1, ..., memberN, valueN} or {value1, ..., valueN}
```


An initializer list has NO type but it may be implicitly converted to a compatible struct (TODO definition of compatible).

If the initializer list is used in a context where its type is needed (eg. no implicit conversion was requested), a new, compatible struct type is created and the initializer list is used to initialize this new type.

## Generic deduction

A function that can deduce its generic parameters takes one of the following syntaxes:
```
function [T] name : (argument : T) -> ReturnType;
function [T] name : (argument : GenericType<T>) -> ReturnType;
function [T] name : (argument : T&) -> ReturnType;
function [T] name : (argument : {member : T, ...}) -> ReturnType;
function [T, N, M] name : (argument : T[N, M]) -> ReturnType;
```
NOTE: pattern matching can be nested.

- A generic type deduction can never require an implicit conversion: the deduced type must be equivalent to the types of the provided expressions.
- When pattern matching, even if two types would be equivalent, type deduction will fail if the provided expression has a type which:
  - Does not at some point alias a `GenericType` specialization and
  - Is not itself a specialization of `GenericType`

- If a type deduction is successful, the function is indistinguishable from a non-generic function with the deduced types manually substituted into the arguments


# Consequences

```rust
let a : ... = ...;

// Is exactly the same as:

typedef __TYPE_OF_A = ...;
let a : __TYPE_OF_A = ...;
// NOTE: any type at any point in the program can be replaced with a unique typedef alias to that type.
```

```c
typedef A : {};
typedef B : {};
typedef C : A;

function takes_A(arg : A) -> void = {};

takes_A(b_obj); // Error: B is a different struct, struct types are never equivalent
takes_A(c_obj); // Good: A and C are the same struct due to the alias
```

```c
function function_with_temp_type : (arg: {a : int, b : int} /* Makes a new struct type */) -> void = {};

let object_1 : {a : int, b : int} /* Makes a new struct type */ = {a : 1, b : 2};
let object_2 = {a : 1, b : 2}; // Initializer list is not implicitly converted to another type and therefore makes a new compatible struct type

function_with_temp_type(object_1); // Error: different struct types are never equivalent
function_with_temp_type(object_2); // Error: different struct types are never equivalent
function_with_temp_type({a : 1, b : 2}); // Good: initializer list implicitly converts to struct type
```

```rust
function[T] function_with_generic_type : (arg: {a : T, b : int}) -> void = {};
// function_with_generic_type can only match against an initializer list since after substituting `T`, any struct would have a different type to the argument (since we create a NEW struct type as the argument's type).
```


```c
typedef Type1<i16> : int;

function[T] foo : (x: Type1<T>) -> T = {...};

const val : int = 2;
foo(val);  // Error: pattern match fails since `int` does not alias a `Type1` specialization at any point -- even though they are equivalent for `Type1<i16>`
```

```c
typedef[Len] string : u8[Len];

function[Len] puts : (str: u8[Len]&) -> int = {...};

const str: string<2>& = "abXX";
puts(&str);  // Good: pattern mach succeeds since `string<2>` aliases `u8[2]&`
```
