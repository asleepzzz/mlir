// RUN: mlir-kevin-driver -kevin-lowerfirs %s | FileCheck %s --check-prefix=LOWERING
// CHECK-LABEL: func @wulala
func @wulala(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
    %0 = kevin.addtest(%arg0, %arg1, %arg2) : f32, f32, f32 to f32
    return %0 : f32
}
