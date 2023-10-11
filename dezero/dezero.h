#ifndef DEZERO_
#define DEZERO_

#define IS_CORE
#ifdef IS_SIMPLE_CORE
#include "core_simple.h"
#include "utils.h"
#endif
#ifdef IS_CORE
#include "core.h"
#include "functions.h"
#include "utils.h"
#endif
#endif