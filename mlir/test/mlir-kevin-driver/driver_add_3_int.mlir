// RUN: mlir-kevin-driver  | FileCheck %s --check-prefix=BEFORE
// RUN: mlir-kevin-driver -multiaddtoadds %s | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-kevin-driver -multiaddtoadds -convert-std-to-llvm %s | FileCheck %s --check-prefix=LLVM_D
// RUN: mlir-kevin-driver -multiaddtoadds -convert-std-to-llvm %s| mlir-translate   -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: mlir-kevin-driver -multiaddtoadds -convert-std-to-llvm %s| mlir-cpu-runner -O3 -e verifyAdd -entry-point-result=void -shared-libs=%kevin_wrapper_library_dir/libmlir_runner_utils%shlibext,%kevin_wrapper_library_dir/libcwrapper%shlibext | FileCheck %s --check-prefix=E2E
// BEFORE: module
// BEFORE: func @kevin_print_f32(f32)
// BEFORE: func @kevin_print_i32(i32)
// BEFORE: func @wulala(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
// BEFORE: %0 = kevin.addtest(%arg0, %arg1, %arg2) : i32, i32, i32 to i32
// BEFORE: return %0 : i32
// BEFORE: }
// BEFORE: func @verifyAdd() {
// BEFORE: %c20_i32 = constant 20 : i32
// BEFORE: %c30_i32 = constant 30 : i32
// BEFORE: %c40_i32 = constant 40 : i32
// BEFORE: %0 = call @wulala(%c20_i32, %c30_i32, %c40_i32) : (i32, i32, i32) -> i32
// BEFORE: call @kevin_print_i32(%0) : (i32) -> ()
// BEFORE: return
// BEFORE: }
// LOWERING: module
// LOWERING: func @kevin_print_f32(f32)
// LOWERING: func @kevin_print_i32(i32)
// LOWERING: func @wulala(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
// LOWERING: %0 = addi %arg0, %arg1 : i32
// LOWERING: %1 = addi %0, %arg2 : i32
// LOWERING: return %1 : i32
// LOWERING: }
// LOWERING: func @verifyAdd() {
// LOWERING: %c20_i32 = constant 20 : i32
// LOWERING: %c30_i32 = constant 30 : i32
// LOWERING: %c40_i32 = constant 40 : i32
// LOWERING: %0 = call @wulala(%c20_i32, %c30_i32, %c40_i32) : (i32, i32, i32) -> i32
// LOWERING: call @kevin_print_i32(%0) : (i32) -> ()
// LOWERING: return
// LOWERING: }
// LLVM_D:module {
// LLVM_D:  llvm.func @kevin_print_f32(!llvm.float)
// LLVM_D:  llvm.func @kevin_print_i32(!llvm.i32)
// LLVM_D:  llvm.func @wulala(%arg0: !llvm.i32, %arg1: !llvm.i32, %arg2: !llvm.i32) -> !llvm.i32 {
// LLVM_D:    %0 = llvm.add %arg0, %arg1 : !llvm.i32
// LLVM_D:    %1 = llvm.add %0, %arg2 : !llvm.i32
// LLVM_D:    llvm.return %1 : !llvm.i32
// LLVM_D:  }
// LLVM_D:  llvm.func @verifyAdd() {
// LLVM_D:    %0 = llvm.mlir.constant(20 : i32) : !llvm.i32
// LLVM_D:    %1 = llvm.mlir.constant(30 : i32) : !llvm.i32
// LLVM_D:    %2 = llvm.mlir.constant(40 : i32) : !llvm.i32
// LLVM_D:    %3 = llvm.call @wulala(%0, %1, %2) : (!llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.i32
// LLVM_D:    llvm.call @kevin_print_i32(%3) : (!llvm.i32) -> ()
// LLVM_D:    llvm.return
// LLVM_D:  }
// LLVM:; ModuleID = 'LLVMDialectModule'
// LLVM:source_filename = "LLVMDialectModule"
// LLVM:declare i8* @malloc(i64)
// LLVM:declare void @free(i8*)
// LLVM:declare void @kevin_print_f32(float)
// LLVM:declare void @kevin_print_i32(i32)
// LLVM:define i32 @wulala(i32 %0, i32 %1, i32 %2) !dbg !3 {
// LLVM:  %4 = add i32 %0, %1, !dbg !7
// LLVM:  %5 = add i32 %4, %2, !dbg !9
// LLVM:  ret i32 %5, !dbg !10
// LLVM:}
// LLVM:define void @verifyAdd() !dbg !11 {
// LLVM:  %1 = call i32 @wulala(i32 20, i32 30, i32 40), !dbg !12
// LLVM:  call void @kevin_print_i32(i32 %1), !dbg !14
// LLVM:  ret void, !dbg !15
// LLVM:}
// E2E:kevin int results is 90
