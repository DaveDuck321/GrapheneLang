@require_once "std/arithmetic.c3"

function main : () -> int = {
    return 1 + 1;
}

/// @COMPILE
/// @GREP_IR !DIFile(directory: "*/tests/debug", filename: "basic.c3")
/// @GREP_IR !DIFile(directory: "*/lib/std", filename: "arithmetic.c3")
/// @GREP_IR distinct !DISubprogram(name: "main", linkageName: "main", scope: !?, file: !?, line: 3, scopeLine: 3, type: !?, unit: !1, spFlags: DISPFlagDefinition)
/// @GREP_IR distinct !DISubprogram(name: "+", linkageName: "__O43__SPECIAL___T_int__T_int_SPECIAL____T_int__T_int", *)
/// @GREP_IR distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !?, producer: "glang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
/// @GREP_IR ret i32 %?, !dbg ![0-9]
/// @GREP_IR !{i32 1, !"Dwarf Version", i32 4}
/// @GREP_IR !{i32 1, !"Debug Info Version", i32 3}
/// @GREP_IR declare void @llvm.dbg.declare(metadata, metadata, metadata)
