set(LLVM_OPTIONAL_SOURCES
  cwrapper.cpp
  mlir-kevin-runner.cpp
  )

include_directories(${MLIR_SOURCE_DIR}/../lld/include)

add_llvm_library(cwrapper SHARED
    cwrapper.cpp
)

target_include_directories(cwrapper
    PRIVATE
    LLVMSupport
)

target_link_libraries(cwrapper
    PUBLIC
    LLVMSupport
)

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    #   lldCommon
    #lldDriver
    #lldELF
    LLVMCore
    LLVMLTO
    LLVMMC
    LLVMMCParser
    LLVMOption
    LLVMSupport
    MLIRJitRunner
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRROCDLIR
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTargetROCDLIR
    MLIRTransforms
    MLIRTranslation
  )

llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens AllTargetsAsmParsers)

  SET(targets_to_link)
  FOREACH(t ${TARGET_LIBS})
    LIST(APPEND targets_to_link "LLVM${t}")
  ENDFOREACH(t)

  add_llvm_tool(mlir-kevin-runner
    mlir-kevin-runner.cpp

    DEPENDS
    cwrapper
    )
  llvm_update_compile_flags(mlir-kevin-runner)
  target_include_directories(mlir-kevin-runner
    PRIVATE
  )
  target_link_libraries(mlir-kevin-runner PRIVATE ${LIBS} ${targets_to_link})







