add_subdirectory(Affine)
add_subdirectory(AVX512)
add_subdirectory(GPU)
add_subdirectory(Linalg)
add_subdirectory(LLVMIR)
add_subdirectory(MIOpen)
add_subdirectory(OpenMP)
add_subdirectory(Quant)
add_subdirectory(SCF)
add_subdirectory(SDBM)
add_subdirectory(Shape)
add_subdirectory(SPIRV)
add_subdirectory(StandardOps)
add_subdirectory(Vector)

set(LLVM_OPTIONAL_SOURCES
  Traits.cpp
)

add_mlir_library(MLIRDialect
  Traits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect

  LINK_LIBS PUBLIC
  MLIRIR
  )
