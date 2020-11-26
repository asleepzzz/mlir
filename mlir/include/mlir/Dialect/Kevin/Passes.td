//===-- Passes.td - MIOpen pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KEVIN_PASSES
#define MLIR_DIALECT_KEVIN_PASSES

include "mlir/Pass/PassBase.td"

def KevinLowerFirstPass : Pass<"kevin-lowerfirst", "ModuleOp"> {
  let summary = "expand convolution into coordinate transformations and gridwise gemm";
  let constructor = "mlir::kevin::createLowerKevinLowerFirstPass()";
}


#endif // MLIR_DIALECT_KEVIN_PASSES
