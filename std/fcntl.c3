// fcntl.h - file control options
// https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/fcntl.h.html

function AT_FDCWD : () -> int = { return -100; } /* Special value used to
                                                    indicate openat should use
                                                    the current working
                                                    directory. */
