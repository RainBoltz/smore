#include "util.h"
#include <direct.h>
#include <sys/stat.h>
#ifndef S_ISDIR
#define S_ISDIR(mode)  (((mode) & S_IFMT) == S_IFDIR)
#endif

bool isDirectory(const char *path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0)
        return 0;
    return S_ISDIR(statbuf.st_mode);
}

