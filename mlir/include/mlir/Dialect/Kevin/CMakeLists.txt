
add_mlir_dialect(KevinOps kevin)
add_mlir_doc(KevinOps -gen-dialect-doc KevinDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRKevinPassIncGen)
