// RUN: mlir-kevin-driver -data_format=float   -data=1,4,6,0.2 | FileCheck %s --check-prefix=BEFORE
// RUN: mlir-kevin-driver -data_format=float   -data=1,4,6,0.2 -multiaddtoadds  | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-kevin-driver -data_format=float   -data=1,4,6,0.2 -multiaddtoadds -convert-std-to-llvm  | FileCheck %s --check-prefix=LLVM_D
// RUN: mlir-kevin-driver -data_format=float   -data=1,4,6,0.2 -multiaddtoadds -convert-std-to-llvm | mlir-translate   -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: mlir-kevin-driver -data_format=float   -data=1,4,6,0.2 -multiaddtoadds -convert-std-to-llvm | mlir-cpu-runner -O3 -e verifyAdd -entry-point-result=void -shared-libs=%kevin_wrapper_library_dir/libmlir_runner_utils%shlibext,%kevin_wrapper_library_dir/libcwrapper%shlibext | FileCheck %s --check-prefix=E2E
// BEFORE: module {
// BEFORE:   func @kevin_print_f32(f32)
// BEFORE:   func @kevin_print_i32(i32)
// BEFORE:   func @wulala(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
// BEFORE:     %0 = kevin.addtest(%arg0, %arg1, %arg2, %arg3) : f32, f32, f32, f32 to f32
// BEFORE:     return %0 : f32
// BEFORE:   }
// BEFORE:   func @verifyAdd() {
// BEFORE:     %cst = constant 1.000000e+00 : f32
// BEFORE:     %cst_0 = constant 4.000000e+00 : f32
// BEFORE:     %cst_1 = constant 6.000000e+00 : f32
// BEFORE:     %cst_2 = constant 2.000000e-01 : f32
// BEFORE:     %0 = call @wulala(%cst, %cst_0, %cst_1, %cst_2) : (f32, f32, f32, f32) -> f32
// BEFORE:     call @kevin_print_f32(%0) : (f32) -> ()
// BEFORE:     return
// BEFORE:   }
// LOWERING: module {
// LOWERING:   func @kevin_print_f32(f32)
// LOWERING:   func @kevin_print_i32(i32)
// LOWERING:   func @wulala(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
// LOWERING:     %0 = addf %arg0, %arg1 : f32
// LOWERING:     %1 = addf %0, %arg2 : f32
// LOWERING:     %2 = addf %1, %arg3 : f32
// LOWERING:     return %2 : f32
// LOWERING:   }
// LOWERING:   func @verifyAdd() {
// LOWERING:     %cst = constant 1.000000e+00 : f32
// LOWERING:     %cst_0 = constant 4.000000e+00 : f32
// LOWERING:     %cst_1 = constant 6.000000e+00 : f32
// LOWERING:     %cst_2 = constant 2.000000e-01 : f32
// LOWERING:     %0 = call @wulala(%cst, %cst_0, %cst_1, %cst_2) : (f32, f32, f32, f32) -> f32
// LOWERING:     call @kevin_print_f32(%0) : (f32) -> ()
// LOWERING:     return
// LOWERING:   }
// LLVM_D:module {
// LLVM_D:  llvm.func @kevin_print_f32(!llvm.float)
// LLVM_D:  llvm.func @kevin_print_i32(!llvm.i32)
// LLVM_D:  llvm.func @wulala(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.float, %arg3: !llvm.float) -> !llvm.float {
// LLVM_D:    %0 = llvm.fadd %arg0, %arg1 : !llvm.float
// LLVM_D:    %1 = llvm.fadd %0, %arg2 : !llvm.float
// LLVM_D:    %2 = llvm.fadd %1, %arg3 : !llvm.float
// LLVM_D:    llvm.return %2 : !llvm.float
// LLVM_D:  }
// LLVM_D:  llvm.func @verifyAdd() {
// LLVM_D:    %0 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
// LLVM_D:    %1 = llvm.mlir.constant(4.000000e+00 : f32) : !llvm.float
// LLVM_D:    %2 = llvm.mlir.constant(6.000000e+00 : f32) : !llvm.float
// LLVM_D:    %3 = llvm.mlir.constant(2.000000e-01 : f32) : !llvm.float
// LLVM_D:    %4 = llvm.call @wulala(%0, %1, %2, %3) : (!llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.float
// LLVM_D:    llvm.call @kevin_print_f32(%4) : (!llvm.float) -> ()
// LLVM_D:    llvm.return
// LLVM_D:  }
// LLVM: ; ModuleID = 'LLVMDialectModule'
// LLVM:source_filename = "LLVMDialectModule"
// LLVM:declare i8* @malloc(i64)
// LLVM:declare void @free(i8*)
// LLVM:declare void @kevin_print_f32(float)
// LLVM:declare void @kevin_print_i32(i32)
// LLVM:define float @wulala(float %0, float %1, float %2, float %3) !dbg !3 {
// LLVM:  %5 = fadd float %0, %1, !dbg !7
// LLVM:  %6 = fadd float %5, %2, !dbg !9
// LLVM:  %7 = fadd float %6, %3, !dbg !10
// LLVM:  ret float %7, !dbg !11
// LLVM:}
// LLVM:define void @verifyAdd() !dbg !12 {
// LLVM:  %1 = call float @wulala(float 1.000000e+00, float 4.000000e+00, float 6.000000e+00, float 0x3FC99999A0000000), !dbg !13
// LLVM:  call void @kevin_print_f32(float %1), !dbg !15
// LLVM:  ret void, !dbg !16
// LLVM:}
// E2E:kevin float results is 11.200000
