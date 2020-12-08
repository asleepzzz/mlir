//===- LowerMIOpenOps.h - MLIR to C++ for MIOpen conversion ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the lowering pass for the MLIR to MIOpen C++ conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KEVIN_LOWERKEVINOPS_H
#define MLIR_DIALECT_KEVIN_LOWERKEVINOPS_H

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Kevin/KevinOps.h"
#include "mlir/Dialect/Kevin/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"


using namespace mlir;
struct KevinAddtestRewritePattern : public OpRewritePattern<kevin::AddtestOp> {
  using OpRewritePattern<kevin::AddtestOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(kevin::AddtestOp op,
                                PatternRewriter &b) const override {
//    auto loc = op.getLoc();
//    auto size = op.getOperands().size();
//    auto numType = op.value().getType();//value is in kevinops.td

//    if (numType.isa<IntegerType>()) {
//              llvm::errs() << "Hello2: "<<size;
	          //auto iter = b.create<ConstantIndexOp>(loc, );

              //for (unsigned i = 2; i < size; i++) {
              //   add = b.create<AddIOp>(loc, add, op.getOperand(i));
              //}
//	auto add = b.create<AddIOp>(loc,op.getOperand(0), op.getOperand(1) );
//	add = b.create<AddIOp>(loc, add, op.getOperand(2));
	//Value added;
        //for (unsigned i = 2; i < size; ++i) {
	//     added = b.create<AddIOp>(loc, add, op.getOperand(i));
//	}

//    } else {
//	     llvm::errs() << "Hello: ";
 //   }

//pattern rewriter will be optimize if no store op ,need to find out 	  
    auto loc = op.getLoc();
    auto operands = op.getOperands();
   

    Value add = b.create<AddIOp>(loc, op.getOperand(0), op.getOperand(1));
    for (unsigned i = 2; i < operands.size(); i++) {
      add = b.create<AddIOp>(loc, add, op.getOperand(i));
    }
    //must do that, so flow operation can use return value
    op.output().replaceAllUsesWith(add);
    op.erase();

  }
};



//===----------------------------------------------------------------------===//
// MovePos lowering.
//===----------------------------------------------------------------------===//

struct KevinMovePosRewritePattern : public OpRewritePattern<kevin::MovePosOp> {
  using OpRewritePattern<kevin::MovePosOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(kevin::MovePosOp op,
                                PatternRewriter &b) const override {

    auto loc = op.getLoc();
    auto memrefType = op.memref().getType().cast<MemRefType>();
    for (unsigned i = 0; i < memrefType.getShape()[0]; ++i) {
      auto iter = b.create<ConstantIndexOp>(loc, i);
      // load
      auto load = b.create<LoadOp>(loc, op.memref(), ValueRange{iter});
      // add
      Value add;
      if (memrefType.getElementType().isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, load, op.getOperand(1 + i));
      } else {
        add = b.create<AddFOp>(loc, load, op.getOperand(1 + i));
      }
      // store
      auto store = b.create<StoreOp>(loc, add, op.memref(), ValueRange{iter});
    }

    op.erase();
    return success();
  }
};





#endif // MLIR_DIALECT_MIOPEN_LOWERMIOPENOPS_H
