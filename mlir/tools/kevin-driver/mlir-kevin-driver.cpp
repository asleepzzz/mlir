//===- mlir-kevin-driver.cpp - MLIR Kevin Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-kevin-driver.
//
//===----------------------------------------------------------------------===//

#include "MlirParse.h"
#include "mlir/Dialect/Kevin/KevinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"



#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static LogicalResult runMLIRPasses(ModuleOp &module, mlir::PassPipelineCLParser &passPipeline, StringRef kernelName) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

    // Passes for lowering Kevin dialect.
    pm.addPass(mlir::kevin::createLowerKevinLowerFirstPass());
    // Passes for lowering linalg dialect.
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createLowerToCFGPass());
pm.addPass(createLowerToLLVMPass());

  return pm.run(module);
}

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR Kevin Dialect driver\n");


llvm::errs() << "=================driver start================" << "\n";



  MLIRContext context;
  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningModuleRef moduleRef;
    module = ModuleOp::create(builder.getUnknownLoc());

  // Determine data type.
  mlir::IntegerType dataType = builder.getI32Type();

  auto funcType =
      builder.getFunctionType({ dataType,dataType}, {});

SmallString<128> kernelName;
kernelName="wulala";
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType);
  module.push_back(func);

  Block *block = func.addEntryBlock();



  MemRefType memrefType = MemRefType::get({2}, dataType);
  AllocOp alloc = builder.create<AllocOp>(builder.getUnknownLoc(), memrefType);

block->push_back(alloc);


    auto zeroConstantI32Op =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), 0, builder.getIntegerType(32));

    auto KPerBlockConstantI32Op =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), 3, builder.getIntegerType(32));

block->push_back(zeroConstantI32Op);
block->push_back(KPerBlockConstantI32Op);

    auto movePosOp = builder.create<kevin::MovePosOp>(

        builder.getUnknownLoc(), alloc,
        ValueRange{zeroConstantI32Op,KPerBlockConstantI32Op}
	);

    block->push_back(movePosOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
  block->push_back(returnOp);

  if (failed(runMLIRPasses(module, passPipeline, kernelName))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }



  module.print(output->os());
  output->keep();



  return 0;
}
