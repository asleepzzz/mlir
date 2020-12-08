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
func @kevin_addtest() -> i32{
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  %deltaZ_i32 = constant 12 : i32
  %deltaT_i32 = kevin.addtest(%deltaY_i32,%deltaX_i32,%deltaZ_i32) : i32, i32,i32 to i32
  return %deltaT_i32 :i32

}

func @kevin_addtest_float(%buffer_f32 : f32) -> f32{
  %deltaY_f32 = constant 16.0 : f32
  %deltaX_f32 = constant 8.0 : f32
  %deltaZ_f32 = constant 12.0 : f32
  %deltaS_f32 = constant 4.0 : f32
  %deltaT_f32 = kevin.addtest(%buffer_f32, %deltaY_f32, %deltaX_f32, %deltaZ_f32, %deltaS_f32) : f32,f32,f32,f32,f32 to f32
  return %deltaT_f32 :f32
}



// CHECK-LABEL: func @kevin_addtest
//   CHECK: kevin.addtest



//func @kevin_addtest(%buffer_i32 : i32) -> i32{
//  %c0 = constant 0 : index
//  %deltaY_i32 = constant 16 : i32
//  %vectorD0 = kevin.addtest(%buffer_i32, %deltaY_i32, %deltaY_i32 )  :   i32, i32, i32
//  return %vectorD0 : i32
//    return
//}



func @miopen_xdlops_gemm_v2_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                       %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> vector<32xf32> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

