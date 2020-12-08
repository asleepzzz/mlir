//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KEVIN_PASSES_H_
#define MLIR_DIALECT_KEVIN_PASSES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"



namespace mlir {
class Pass;

namespace kevin {
std::unique_ptr<Pass> createLowerKevinLowerFirstPass();
std::unique_ptr<OperationPass<FuncOp>> createMultiAddTransPass();
} // namespace miopen
} 



#endif // MLIR_DIALECT_KEVIN_PASSES_H_
