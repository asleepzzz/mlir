// RUN: mlir-kevin-driver -kevin-lowerfirst  | FileCheck %s
// CHECK-LABEL: module
// CHECK-NEXT: func @kevin_print_f32(f32)
// CHECK-NEXT: func @kevin_print_i32(i32)
// CHECK-NEXT: func @wulala(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
// CHECK-NEXT: %0 = addi %arg0, %arg1 : i32
// CHECK-NEXT: %1 = addi %0, %arg2 : i32
// CHECK-NEXT: return %1 : i32
// CHECK-NEXT: }
// CHECK-NEXT: func @verifyAdd() {
// CHECK-NEXT: %c20_i32 = constant 20 : i32
// CHECK-NEXT: %c30_i32 = constant 30 : i32
// CHECK-NEXT: %c40_i32 = constant 40 : i32
// CHECK-NEXT: %0 = call @wulala(%c20_i32, %c30_i32, %c40_i32) : (i32, i32, i32) -> i32
// CHECK-NEXT: call @kevin_print_i32(%0) : (i32) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
