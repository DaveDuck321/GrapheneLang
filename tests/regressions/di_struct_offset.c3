typedef S : { x : i32, y : i64 }

function main : () -> int = {
    let s : S = {1, 2};

    return 0;
}

/// @FOR x86_64_linux
/// @COMPILE
/// @GREP_IR !DIDerivedType(baseType: !?, name: "x", offset: 0, size: 32, tag: DW_TAG_member)
/// @GREP_IR !DIDerivedType(baseType: !?, name: "y", offset: 64, size: 64, tag: DW_TAG_member)
