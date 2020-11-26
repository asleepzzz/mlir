file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRKevinOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Kevin

  DEPENDS
  MLIRKevinOpsIncGen
  )
target_link_libraries(MLIRKevinOps
  PUBLIC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )

add_subdirectory(Transforms)
