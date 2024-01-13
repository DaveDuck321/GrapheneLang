// sys/stat.h - data returned by the stat() function
// https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/stat.h.html
@require_once "../wrappers.c3"
@require_once "syscalls.c3"
@require_once "types.c3"

// This is VERY difficult to get right. fstat takes a `stat_t` buffer BUT we
// don't actually know the layout of `stat_t` because it depends on the kernel's
// compile flags. POSIX just guarantees that the struct contains a minimum
// subset of fields.

// https://man7.org/linux/man-pages/man3/stat.3type.html
// The below structs are generated via the following command:
// echo "#include <sys/stat.h>" | g++ -xc -E -

typedef timespec : {
  tv_sec : time_t,
  tv_nsec : isize,
}

typedef stat_t : {
    st_dev : dev_t,
    st_ino : ino_t,
    st_nlink : nlink_t,
    st_mode : mode_t,
    st_uid : uid_t,
    st_gid : gid_t,
    __pad0 : int,
    st_rdev : dev_t,
    st_size : off_t,
    st_blksize : blksize_t,
    st_blocks : blkcnt_t,
    st_atim : timespec,
    st_mtim : timespec,
    st_ctim : timespec,
    __glibc_reserved : isize[3],
}

function stat : ( fd : int ) -> Optional<stat_t> = {
    // TODO: propagate the actual error code
    mut stat_buf : stat_t;
    let result : int = sys_fstat(fd, &reinterpret_addr<u8[&]>(&mut stat_buf));
    if result < 0 {
        return make<Optional<stat_t>>();
    }
    return make<Optional<stat_t>>(stat_buf);
}
