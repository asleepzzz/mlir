//===- XdlopsCodeSelection.h - MLIR to C++ for MIOpen conversion ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code selection logic for XDLOPS instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_XDLOPS_CODE_SELECTION_H
#define MLIR_XDLOPS_CODE_SELECTION_H

using namespace mlir;

//===----------------------------------------------------------------------===//
// XDLOPS code selection logic.
//===----------------------------------------------------------------------===//
struct XdlopsCodeSelection {
  StringRef mfmaInstr;
  int64_t MPerXdlops;
  int64_t NPerXdlops;
  int64_t MRepeats;
  int64_t NRepeats;

  int64_t group_size;
  int64_t num_groups_blk;
  int64_t num_regs_blk;
  int64_t num_threads_blk;
  int64_t wave_size;
  int64_t num_input_blks;
  int64_t num_output_blks;
  int64_t num_regs_xdlops;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t cycles;
  int64_t k_base;

  static XdlopsCodeSelection get(FloatType dataType, int64_t MPerWave, int64_t NPerWave, OpBuilder &b) {
    // Determine which XDLOPS be used.
    int64_t MPerXdlops = 0, NPerXdlops = 0, MRepeats = 0, NRepeats = 0;
    StringRef mfmaInstr = "";
    if (dataType == b.getF32Type()) {
      if (MPerWave == 128 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 2;
      } else if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x1xf32";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x1xf32";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x2xf32";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_16x16x1xf32";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x1xf32";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x4xf32";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 8 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x1xf32";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x1xf32";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 4 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x1xf32";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x1xf32";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else {
        llvm::errs() << "Unsupported case:\n";
        //llvm::errs() << "M, N, K:" << M << " " << N << " " << K << "\n";
        llvm::errs() << "MPerWave: " << MPerWave << "\n";
        llvm::errs() << "NPerWave: " << NPerWave << "\n";
        llvm::errs() << "dataType: ";
        dataType.dump();
        llvm::errs() << "\n";
      }
    } else if (dataType == b.getF16Type()) {
      if (MPerWave == 128 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 2;
      } else if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x4f16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x8f16";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x4f16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x16f16";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 8 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 4 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x4f16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else {
        llvm::errs() << "Unsupported case:\n";
        //llvm::errs() << "M, N, K:" << M << " " << N << " " << K << "\n";
        llvm::errs() << "MPerWave: " << MPerWave << "\n";
        llvm::errs() << "NPerWave: " << NPerWave << "\n";
        llvm::errs() << "dataType: ";
        dataType.dump();
        llvm::errs() << "\n";
      }
    } else if (dataType == b.getBF16Type()) {
      if (MPerWave == 128 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 2;
      } else if (MPerWave == 128 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 128 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 2;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 64 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 64 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 64;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 32 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_32x32x2bf16";
        MPerXdlops = 32;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 32 && NPerWave == 32) {
        mfmaInstr = "mfma_f32_32x32x4bf16";
        MPerXdlops = 32;
        NPerXdlops = 32;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 16 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_16x16x2bf16";
        MPerXdlops = 16;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 16 && NPerWave == 16) {
        mfmaInstr = "mfma_f32_16x16x8bf16";
        MPerXdlops = 16;
        NPerXdlops = 16;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 8 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 8 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 8;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else if (MPerWave == 4 && NPerWave == 128) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 2;
      } else if (MPerWave == 4 && NPerWave == 64) {
        mfmaInstr = "mfma_f32_4x4x2bf16";
        MPerXdlops = 4;
        NPerXdlops = 64;
        MRepeats = 1;
        NRepeats = 1;
      } else {
        llvm::errs() << "Unsupported case:\n";
        //llvm::errs() << "M, N, K:" << M << " " << N << " " << K << "\n";
        llvm::errs() << "MPerWave: " << MPerWave << "\n";
        llvm::errs() << "NPerWave: " << NPerWave << "\n";
        llvm::errs() << "dataType: ";
        dataType.dump();
        llvm::errs() << "\n";
      }
    }

    // Obtain properties of MFMA instructions.
    int64_t group_size, num_groups_blk, num_regs_blk, num_threads_blk, wave_size, num_input_blks, num_output_blks, num_regs_xdlops, m, n, k, cycles, k_base;
    if (mfmaInstr == "mfma_f32_32x32x1xf32") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 1;
      cycles          = 64;
      k_base          = 1;
    } else if (mfmaInstr == "mfma_f32_32x32x2xf32") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 2;
      cycles          = 64;
      k_base          = 1;
    } else if (mfmaInstr == "mfma_f32_16x16x4xf32") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 4;
      cycles          = 32;
      k_base          = 1;
    } else if (mfmaInstr == "mfma_f32_16x16x1xf32") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 1;
      cycles          = 32;
      k_base          = 1;
    } else if (mfmaInstr == "mfma_f32_4x4x1xf32") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 64;
      wave_size       = 64;
      num_input_blks  = 1;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m               = 4;
      n               = 64;
      k               = 1;
      cycles          = 8;
      k_base          = 1;
    } else if (mfmaInstr == "mfma_f32_32x32x4f16") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 4;
      cycles          = 64;
      k_base          = 4;
    } else if (mfmaInstr == "mfma_f32_32x32x8f16") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 8;
      cycles          = 64;
      k_base          = 4;
    } else if (mfmaInstr == "mfma_f32_16x16x16f16") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 16;
      cycles          = 32;
      k_base          = 4;
    } else if (mfmaInstr == "mfma_f32_16x16x4f16") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 4;
      cycles          = 32;
      k_base          = 4;
    } else if (mfmaInstr == "mfma_f32_4x4x4f16") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 64;
      wave_size       = 64;
      num_input_blks  = 1;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m               = 4;
      n               = 64;
      k               = 4;
      cycles          = 8;
      k_base          = 4;
    } else if (mfmaInstr == "mfma_f32_32x32x2bf16") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 2;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 2;
      cycles          = 64;
      k_base          = 2;
    } else if (mfmaInstr == "mfma_f32_32x32x4bf16") {
      group_size      = 4;
      num_groups_blk  = 4;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 32;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 32;
      n               = 32;
      k               = 4;
      cycles          = 64;
      k_base          = 2;
    } else if (mfmaInstr == "mfma_f32_16x16x8bf16") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 1;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 8;
      cycles          = 32;
      k_base          = 2;
    } else if (mfmaInstr == "mfma_f32_32x32x2xf32") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 16;
      wave_size       = 64;
      num_input_blks  = wave_size / num_threads_blk;
      num_output_blks = 4;
      num_regs_xdlops = num_regs_blk * num_output_blks;
      m               = 16;
      n               = 16;
      k               = 2;
      cycles          = 32;
      k_base          = 2;
    } else if (mfmaInstr == "mfma_f32_4x4x2bf16") {
      group_size      = 4;
      num_groups_blk  = 1;
      num_regs_blk    = group_size * num_groups_blk;
      num_threads_blk = 64;
      wave_size       = 64;
      num_input_blks  = 1;
      num_output_blks = 1;
      num_regs_xdlops = 4;
      m               = 4;
      n               = 64;
      k               = 2;
      cycles          = 8;
      k_base          = 2;
    } else {
      llvm::errs() << "Unsupported case as mfmaInstr not selected!\n";
    }

    // Populate result.
    XdlopsCodeSelection result;
    result.mfmaInstr = mfmaInstr;
    result.MPerXdlops = MPerXdlops;
    result.NPerXdlops = NPerXdlops;
    result.MRepeats = MRepeats;
    result.NRepeats = NRepeats;

    result.group_size = group_size;
    result.num_groups_blk = num_groups_blk;
    result.num_regs_blk = num_regs_blk;
    result.num_threads_blk = num_threads_blk;
    result.wave_size = wave_size;
    result.num_input_blks = num_input_blks;
    result.num_output_blks = num_output_blks;
    result.num_regs_xdlops = num_regs_xdlops;
    result.m = m;
    result.n = n;
    result.k = k;
    result.cycles = cycles;
    result.k_base = k_base;

    return result;
  }
};

#endif
