// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @kevin_addtest_float
// CHECK-NEXT: kevin.addtest
func @kevin_addtest_float(%arg0: f32, %arg1: f32) -> f32{
  %0 = kevin.addtest(%arg0, %arg1) : f32, f32 to f32
  return %0 : f32
}

// CHECK-LABEL: func @kevin_addtest_int
// CHECK-NEXT: kevin.addtest
func @kevin_addtest_int(%arg0: i32, %arg1: i32, %arg2: i32) -> i32{
  %0 = kevin.addtest(%arg0, %arg1, %arg2) : i32, i32, i32 to i32
  return %0 : i32
}

