// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @kevin_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  kevin.conv2d(%filter, %input, %output) {
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["n", "c", "hi", "wi"],
    output_layout = ["n", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @kevin_conv2d
// CHECK-NEXT: kevin.conv2d

//memref<1xi32>
func @kevin_add(%buffer_i32 : memref<1xi32>) {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  %deltaZ_i32 = constant 12 : i32
  kevin.add(%buffer_i32, %deltaY_i32, %deltaX_i32, %deltaZ_i32) : memref<1xi32>

  return
}
// CHECK-LABEL: func @kevin_add
//   CHECK: kevin.add


//memref<1xi32>
func @kevin_addtest(%buffer_i32 : i32) {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  %deltaZ_i32 = constant 12 : i32
  kevin.addtest(%buffer_i32, %deltaY_i32, %deltaX_i32, %deltaZ_i32) : i32

  return
}
// CHECK-LABEL: func @kevin_addtest
//   CHECK: kevin.addtest

