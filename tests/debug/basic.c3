@require_once "std/arithmetic.c3"

function main : () -> int = {
    return 1 + 1;
}

/// @COMPILE
/// @GREP_IR !DIFile(filename: "basic.c3",
/// @GREP_IR !DIFile(filename: "arithmetic.c3",
/// @GREP_IR distinct !DISubprogram(name: "main", linkageName: "main"
/// @GREP_IR distinct !DISubprogram(name: "+", linkageName: "__O43__SPECIAL___T_int__T_int_SPECIAL____T_int__T_int"
/// @GREP_IR distinct !DICompileUnit(language: DW_LANG_C,
/// @GREP_IR !dbg !
/// @GREP_IR scopeLine: 3
