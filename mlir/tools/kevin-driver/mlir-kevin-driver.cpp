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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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


static cl::opt<std::string> data_format("data_format",
                                         cl::desc("int or float"),
                                         cl::value_desc("int or float"),
                                         cl::init("int"));

static cl::opt<std::string> data("data",
                                         cl::desc("inpiut data"),
                                         cl::value_desc("input data"),
                                         cl::init("20,30,40"));

static cl::opt<bool> loweringDefault(
    "c", cl::desc("To lower with default pipeline"),
    cl::value_desc("To lower with default pipeline"), cl::init(false));

static LogicalResult runMLIRPasses(ModuleOp &module, mlir::PassPipelineCLParser &passPipeline, StringRef kernelName) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  if (loweringDefault.getValue()) {
    // Passes for lowering Kevin dialect.
    pm.addPass(mlir::kevin::createMultiAddTransPass());//-multiaddtoadds
//    pm.addPass(mlir::kevin::createLowerKevinLowerFirstPass());//--kevin-lowerfirst
    // Passes for lowering linalg dialect.
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createLowerToCFGPass());
      pm.addPass(createLowerToLLVMPass());
  }else {
    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm)))
      return failure();
  }



  return pm.run(module);
}


std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR Kevin Dialect driver\n");

  std::vector<std::string> parameters = split(data, ',');
  if(parameters.size() < 2){
    llvm::errs() << "can not add" << "\n";
    exit(1);
  }


llvm::errs() << "=================driver start ,data size is " << parameters.size()<<"================\n";



  MLIRContext context;
  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningModuleRef moduleRef;
    module = ModuleOp::create(builder.getUnknownLoc());

//APFloat kevinbf16test(APFloat::BFloat(), "1.2");

  // Determine data type.
  mlir::IntegerType dataType = builder.getI32Type();

    auto kevinprintf32FuncOp = FuncOp::create(
      builder.getUnknownLoc(), "kevin_print_f32",
      builder.getFunctionType(
          {builder.getF32Type()}, {}));
    module.push_back(kevinprintf32FuncOp);
               
    auto kevinprinti32FuncOp = FuncOp::create(
      builder.getUnknownLoc(), "kevin_print_i32",
      builder.getFunctionType(
          {builder.getI32Type()}, {}));
    module.push_back(kevinprinti32FuncOp);
 


  auto funcType = builder.getFunctionType({ }, {});
llvm::SmallVector<mlir::Type,2> functiontypes;


    if(data_format == "float")
    {
        for(unsigned i = 0; i < parameters.size(); i++){
            functiontypes.push_back(builder.getF32Type());
        }

    } else 
    {
	for(unsigned i = 0; i < parameters.size(); i++){
            functiontypes.push_back(builder.getI32Type());
        }
    }
 


funcType = builder.getFunctionType({functiontypes }, {functiontypes[0]});


SmallString<128> kernelName;
kernelName="wulala";
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType);
  module.push_back(func);

  Block *block = func.addEntryBlock();
  auto args = block->getArguments();


    auto addTestOp = builder.create<kevin::AddtestOp>(
        builder.getUnknownLoc(), functiontypes[0],
	ValueRange{args}
        );

    block->push_back(addTestOp);



//below is not multi add  
/*
  MemRefType memrefType = MemRefType::get({2}, dataType);
  AllocOp alloc = builder.create<AllocOp>(builder.getUnknownLoc(), memrefType);

block->push_back(alloc);








//memset start

SmallVector<int64_t, 1> tesnordim;
tesnordim.push_back(2);
//tesnordim.push_back(5);


  auto kevinMemRefType = MemRefType::get(
      ArrayRef<int64_t>(tesnordim.begin(), tesnordim.end()), dataType);
  auto UnknownSizeMemRefType =
      MemRefType::get({-1}, dataType);

  auto intAllocOp =
      builder.create<AllocOp>(builder.getUnknownLoc(), kevinMemRefType);
  block->push_back(intAllocOp);


      
         auto castop = builder.create<MemRefCastOp>(
               builder.getUnknownLoc(), intAllocOp, UnknownSizeMemRefType);
                 block->push_back(castop);
      

  auto memset2DFuncOp = FuncOp::create(
      builder.getUnknownLoc(), "memset1DIntt",
      builder.getFunctionType(
          {UnknownSizeMemRefType, dataType}, {}));
  module.push_back(memset2DFuncOp);
		 

    auto CpuMemsetOp = builder.create<CallOp>(
      builder.getUnknownLoc(), memset2DFuncOp,
      ValueRange{castop, threeConstantI32Op});
  block->push_back(CpuMemsetOp);
//memset over




    auto movePosOp = builder.create<kevin::MovePosOp>(
        builder.getUnknownLoc(), intAllocOp,
        ValueRange{zeroConstantI32Op,threeConstantI32Op}
	);

    block->push_back(movePosOp);



//printf start


		 
  auto forprintfFuncOp = FuncOp::create(
      builder.getUnknownLoc(), "forprintf",
      builder.getFunctionType(
          {UnknownSizeMemRefType, dataType}, {}));
  module.push_back(forprintfFuncOp);


    auto printOp = builder.create<CallOp>(
      builder.getUnknownLoc(), forprintfFuncOp,
      ValueRange{castop, threeConstantI32Op});
  block->push_back(printOp);
  
//printf over


  */





//multiAddOp


  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{addTestOp});
  block->push_back(returnOp);





  //call wulala
  {
    auto mainType = builder.getFunctionType({}, {});
    auto main = FuncOp::create(builder.getUnknownLoc(), "verifyAdd", mainType);
    module.push_back(main);
    Block *secondBlock = main.addEntryBlock();
    llvm::SmallVector<mlir::Value,2> parameters_of_call_wulala;


    for(unsigned i = 0; i < parameters.size(); i++){
        if (data_format == "float") {	    
            auto constantData = builder.create<ConstantFloatOp>(
            builder.getUnknownLoc(), APFloat((float)std::atof(parameters[i].c_str())), builder.getF32Type());
            parameters_of_call_wulala.push_back(constantData);
            secondBlock->push_back(constantData);

        } else {
	    auto constantData = builder.create<ConstantIntOp>(
            builder.getUnknownLoc(), (int)std::atoi(parameters[i].c_str()),builder.getIntegerType(32));
            parameters_of_call_wulala.push_back(constantData);
	    secondBlock->push_back(constantData);
        }
    }




    auto callWulalaOp =
        builder.create<CallOp>(builder.getUnknownLoc(), func,
                               parameters_of_call_wulala);

    secondBlock->push_back(callWulalaOp);

    //print
    
    if (data_format == "float") {
      auto callPrintfOp = builder.create<CallOp>(
      builder.getUnknownLoc(), kevinprintf32FuncOp,
      ValueRange{callWulalaOp.getResults()});
      secondBlock->push_back(callPrintfOp);
    } else {
      auto callPrintfOp = builder.create<CallOp>(
      builder.getUnknownLoc(), kevinprinti32FuncOp,
      ValueRange{callWulalaOp.getResults()});
      secondBlock->push_back(callPrintfOp);
    }

    auto mainReturnOp =
        builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
    secondBlock->push_back(mainReturnOp);

  }




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
