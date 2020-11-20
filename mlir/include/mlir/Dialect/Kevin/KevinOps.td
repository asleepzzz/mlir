//===- KevinOps.td - Kevin operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Kevin operations.
//
//===----------------------------------------------------------------------===//

#ifndef KEVIN_OPS
#define KEVIN_OPS

include "mlir/IR/OpBase.td"
//include "mlir/Transforms/LoopLikeInterface.td"

def Kevin_Dialect : Dialect {
  let name = "kevin";
  let cppNamespace = "kevin";
}

// Base class for Kevin dialect ops.
class Kevin_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Kevin_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
  }];
}

def Kevin_AddOp :
    Kevin_Op<"add">,
    Arguments<(ins AnyMemRef:$memref,  Variadic<AnyTypeOf<[AnyInteger]>>:$values)> {
    let summary = "Add values to indices within the memref.";
    let description = [{
      The `kevin.add` op adds values to indices within the memref.
    }];
}

def Kevin_AddtestOp :
    Kevin_Op<"addtest">,
    Arguments<(ins AnyTypeOf<[AnyInteger]>:$value,  Variadic<AnyTypeOf<[AnyInteger]>>:$values)> {
    let summary = "Add values to indices within the memref.";
    let description = [{
      The `kevin.add` op adds values to indices within the memref.
    }];
}


def Kevin_Conv2DOp :
    Kevin_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32], [4]>:$filter,
                   MemRefRankOf<[F32], [4]>:$input,
                   MemRefRankOf<[F32], [4]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `kevin.conv2d` op computes 2D convolution forward.
  }];
}

def Kevin_Conv2DBwdDataOp :
    Kevin_Op<"conv2d_bwd_data">,
    Arguments<(ins MemRefRankOf<[F32], [4]>:$filter,
                   MemRefRankOf<[F32], [4]>:$input,
                   MemRefRankOf<[F32], [4]>:$output)> {
  let summary = "2D convolution backward data";
  let description = [{
    The `kevin.conv2d` op computes 2D convolution backward data.
  }];
}

def Kevin_TransformOp :
    Kevin_Op<"transform">,
    Arguments<(ins AnyMemRef:$input)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    The `kevin.transform` op transforms tensor coordinates.
  }];
}

def Kevin_GridwiseGemmOp :
    Kevin_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>:$filter,
                   MemRefRankOf<[F32], [2]>:$input,
                   MemRefRankOf<[F32], [2]>:$output)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `kevin.gridwise_gemm` op computes gridwise GEMM.
  }];
}

// gridwise_gemm_ex would eventually be replaced with gridwise_gemm
// keep it as a separate op for now.
// the difference with gridwise_gemm is gridwise_gemm_ex carries tuning
// parameters. On the other hand, gridwise_gemm only carries convolution
// configuration.
def Kevin_GridwiseGemmExOp:
    Kevin_Op<"gridwise_gemm_ex">,
    Arguments<(ins MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>)> {
  let summary = "Gridwise GEMM extended";
  let description = [{
    The `kevin.gridwise_gemm_ex` op computes gridwise GEMM, with tuning
    parameters supplied in the attributes.
  }];
}

// Memory allocation on GPU memory hierachy.
def Kevin_GpuAllocOp:
    Kevin_Op<"alloc">,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Memory allocation on GPU";
  let description = [{
    The `kevin.alloc` op allocates memory on GPU.
    - Address space 0 : global.
    - Address space 3 : LDS.
    - Address space 5 : private (VGPR).
    All other values would be considered as allocation on global.
  }];
}

// subview provides a subsection view of a memref.
def Kevin_SubviewOp:
    Kevin_Op<"subview">,
    Arguments<(ins AnyMemRef:$input,
                   Index:$offset)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Subview of memref on GPU";
  let description = [{
    The `kevin.subview` op creates a subview of a memref on GPU.
    The memref must be on either LDS (address space 3) or private (address
    space 5).
  }];
}

// TBD: eventually replace this with linalg.fill?
def Kevin_FillOp:
    Kevin_Op<"fill">,
    Arguments<(ins AnyMemRef:$input,
                   AnyTypeOf<[AnyInteger, AnyFloat]>:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `kevin.fill` op fills a memref on GPU with a constant value.
  }];
}

def Kevin_MovePosOp:
    Kevin_Op<"move_pos">,
    Arguments<(ins AnyMemRef:$memref,
                   Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$values)> {
    let summary = "Add values to indices within the memref.";
    let description = [{
      The `kevin.fill` op adds values to indices within the memref.
    }];
}

def Kevin_WorkgroupBarrierOp:
    Kevin_Op<"workgroup_barrier"> {
  let summary = "Setup an workgroup barrier";
  let description = [{
    The `kevin.workgroup_barrier` op sets up a workgroup-level barrier.
  }];
}

def Kevin_WorkgroupIdOp:
    Kevin_Op<"workgroup_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workgroup ID";
  let description = [{
    The `kevin.workgroup_id` op gets the current workgroup ID.
  }];
}

def Kevin_WorkitemIdOp:
    Kevin_Op<"workitem_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workitem ID";
  let description = [{
    The `kevin.workgroup_id` op gets the current workitem ID.
  }];
}

// blockwise_copy
def Kevin_BlockwiseCopyOp:
    Kevin_Op<"blockwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   AnyMemRef:$sourceCoord,
                   AnyMemRef:$destCoord,
                   Optional<AnyMemRef>:$buffer)> {
  let summary = "Blockwise GPU data movement";
  let description = [{
    The `kevin.blockwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to LDS (naive tensor).
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
  }];
}

// threadwise_copy
def Kevin_ThreadwiseCopyOp:
    Kevin_Op<"threadwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   Variadic<I32>:$sourceAndDestCoord)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `kevin.threadwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
    - LDS (naive tensor) to register (naive tensor).
    - Register (naive tensor) to global (generic tensor).
  }];
}

// blockwise_gemm
def Kevin_BlockwiseGemmOp:
    Kevin_Op<"blockwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>:$matrixA,
                   MemRefRankOf<[F32], [2]>:$matrixB,
                   MemRefRankOf<[F32], [2]>:$matrixC,
                   Index:$threadOffsetA,
                   Index:$threadOffsetB)> {
  let summary = "Blockwise GEMM";
  let description = [{
    The `kevin.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall reside on register (naive tensor).
  }];
}

// threadwise_gemm
def Kevin_ThreadwiseGemmOp:
    Kevin_Op<"threadwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>:$matrixA,
                   MemRefRankOf<[F32], [2]>:$matrixB,
                   MemRefRankOf<[F32], [2]>:$matrixC)> {
  let summary = "Threadwise GEMM";
  let description = [{
    The `kevin.threadwise_gemm` op does GEMM at thread level.
    - Matrix A and Matrix B shall reside on register (naive tensor).
    - Matrix C shall reside on LDS (naive tensor).
  }];
}

#endif // KEVIN_OPS
