add_mlir_dialect_library(MLIRKevinTransforms
	LowerKevinOps.cpp
	MultAddTransToAdds.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRKevinPassIncGen
  MLIRSCFToStandard
  MLIRStandardOps
)

target_link_libraries(MLIRKevinTransforms
  PUBLIC
  MLIRAffineToStandard
  MLIRIR
  MLIRLLVMIR
  MLIRKevinOps
  MLIRPass
  MLIRSCFToStandard
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )
