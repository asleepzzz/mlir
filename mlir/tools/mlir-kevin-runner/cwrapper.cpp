#include <cassert>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

extern "C" void kevin_print_f32(float f) { printf("kevin float results is %f\n", f); }
extern "C" void kevin_print_i32(int d) { printf("kevin int results is %d\n", d); }

extern "C" void memset1DIntt(int *allocated, int *aligned,
                                  int64_t offset, int64_t size0,
                                  int64_t stride0, 
                                  int value) {
  for (unsigned i = 0; i < size0; ++i)
      aligned[i * stride0 ] = value+i;
  printf("===hey %d==\n",aligned[0]);
  printf("===hey %d==\n",aligned[1]);
}


extern "C" void memset2DIntt(int *allocated, int *aligned,
                                  int64_t offset, int64_t size0, int64_t size1,
                                  int64_t stride0, int64_t stride1,
                                  int value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      aligned[i * stride0 + j * stride1] = value+j;
  printf("===hey %d==\n",aligned[3]);
}


extern "C" void forprintf(int *allocated, int *aligned,
                                  int64_t offset, int64_t size0,
                                  int64_t stride0,
                                  int value) {
  //for (unsigned i = 0; i < size0; ++i)
  //  for (unsigned j = 0; j < size1; ++j)
  //    aligned[i * stride0 + j * stride1] = value+j;
  printf("===hey end %d==\n",aligned[0]);
  printf("===hey end %d==\n",aligned[1]);
}



