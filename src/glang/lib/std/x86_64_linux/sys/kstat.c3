// https://elixir.bootlin.com/linux/latest/source/arch/x86/include/uapi/asm/stat.h
typedef stat_t : {
    st_dev : isize,
    st_ino : isize,
    st_nlink : isize,
    st_mode : u32,
    st_uid : u32,
    st_gid : u32,
    __pad0 : u32,
    st_rdev : isize,
    st_size : isize,
    st_blksize : isize,
    st_blocks : isize,
    st_atime : isize,
    st_atime_nsec : isize,
    st_mtime : isize,
    st_mtime_nsec : isize,
    st_ctime : isize,
    st_ctime_nsec : isize,
    __unused : isize[3],
}
