@require_once "circular_imports.c3";

/// @COMPILE; EXPECT ERR
/// File '*/circular_imports.c3', line 1, in '@require_once "circular_imports.c3"'
/// Error: cannot import '*/circular_imports.c3' since this would be a circular dependency: already imported from parent '<compile target>'
