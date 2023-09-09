// fcntl.h - file control options
// https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/fcntl.h.html

function O_RDONLY : () -> u32 = { return 0x00; }
function O_WRONLY : () -> u32 = { return 0x01; }
function O_RDWR : () -> u32 = { return 0x02; }

function AT_FDCWD : () -> int = { return -100; } /* Special value used to
                                                    indicate openat should use
                                                    the current working
                                                    directory. */
