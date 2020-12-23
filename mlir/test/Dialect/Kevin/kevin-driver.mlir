// RUN: mlir-opt %s | FileCheck %s
// Run: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s
// this part generated from ./bin/mlir-kevin-driver -data_format=int -data=1,4,6,2
// and  ./bin/mlir-kevin-driver -data_format=float


func @kevin_print_f32(f32)
func @kevin_print_i32(i32)

// CHECK-LABEL: func @wulala
// CHECK-SAME: (%arg0: f32, %arg1: f32, %arg2: f32) -> f32
func @wulala(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
// CHECK: kevin.addtest
  %0 = kevin.addtest(%arg0, %arg1, %arg2) : f32, f32, f32 to f32
  return %0 : f32
}

// CHECK-LABEL: func @verifyAdd
func @verifyAdd() {
    %cst = constant 2.000000e+01 : f32
    %cst_0 = constant 3.000000e+01 : f32
    %cst_1 = constant 4.000000e+01 : f32
// CHECK: call @wulala
    %0 = call @wulala(%cst, %cst_0, %cst_1) : (f32, f32, f32) -> f32
    call @kevin_print_f32(%0) : (f32) -> ()
    return
}

// CHECK-LABEL: func @wulala2
// CHECK-SAME: (%arg0: i32, %arg1: i32) -> i32
func @wulala2(%arg0: i32, %arg1: i32) -> i32 {
    %0 = kevin.addtest(%arg0, %arg1) : i32, i32 to i32
    return %0 : i32
}

// CHECK-LABEL: func @verifyAdd2
func @verifyAdd2() {
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %0 = call @wulala2(%c1_i32, %c2_i32) : (i32, i32) -> i32
// CHECK: call @kevin_print_i32
    call @kevin_print_i32(%0) : (i32) -> ()
    return
}

