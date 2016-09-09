#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "BilinearSamplerBHWD.cu"
#include "Huber.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcugvnn(lua_State *L);

int luaopen_libcugvnn(lua_State *L)
{
  lua_newtable(L);
  cunn_BilinearSamplerBHWD_init(L);
  cunn_HuberCriterion_init(L);  

  return 1;
}
