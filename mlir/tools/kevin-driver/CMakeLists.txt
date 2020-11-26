set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRKevinOps
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRSDBM
  MLIRStandardOps
  MLIRTransforms
  MLIRJitRunner
  MLIRSupport
  MLIRIR

  MLIREDSC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRTargetLLVMIR

)

add_llvm_executable(mlir-kevin-driver
  PARTIAL_SOURCES_INTENDED

  mlir-kevin-driver.cpp

  DEPENDS
  ${LIBS}
)

llvm_update_compile_flags(mlir-kevin-driver)
target_link_libraries(mlir-kevin-driver PRIVATE ${LIBS})
mlir_check_link_libraries(mlir-kevin-driver)



