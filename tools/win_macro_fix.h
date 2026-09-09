// Workaround: new Windows SDK defines `small` (rpcndr.h: #define small char),
// which breaks CUDA 11.8's cub headers (TempStorage small[...]).
// Pre-include the offending headers once, then remove the poisonous macros.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef small
#undef near
#undef far
#undef min
#undef max
#endif
