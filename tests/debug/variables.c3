typedef [T] S : { ptr : S<T>&, x : T, y : f32 }

function main : () -> int = {
    mut s : S<int>;
    return 0;
}

/// @COMPILE
/// @GREP_IR !DICompositeType(elements: !??, name: "S<int>", size: 128, tag: DW_TAG_structure_type)
/// @GREP_IR !DIDerivedType(baseType: !?, size: 64, tag: DW_TAG_reference_type)
/// @GREP_IR !DIDerivedType(baseType: !?, name: "ptr", offset: 0, size: 64, tag: DW_TAG_member)
/// @GREP_IR !DIBasicType(encoding: DW_ATE_signed, name: "int", size: 32, tag: DW_TAG_base_type)
/// @GREP_IR !DIDerivedType(baseType: !?, name: "x", offset: 64, size: 32, tag: DW_TAG_member)
/// @GREP_IR !DIBasicType(encoding: DW_ATE_float, name: "f32", size: 32, tag: DW_TAG_base_type)
/// @GREP_IR !DIDerivedType(baseType: !9, name: "y", offset: 96, size: 32, tag: DW_TAG_member)
/// @GREP_IR !{!?, !?, !??}
/// @GREP_IR !DILocalVariable(file: !?, line: 4, name: "s", scope: !?, type: !?)
