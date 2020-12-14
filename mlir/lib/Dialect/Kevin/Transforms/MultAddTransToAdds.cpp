#include "PassDetail.h"
#include "mlir/Dialect/Kevin/KevinOps.h"
#include "mlir/Dialect/Kevin/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct MultAddTransToAdds : public MultAddTransToAddsBase<MultAddTransToAdds> {
  void runOnFunction() override;
};

} // end anonymous namespace

void MultAddTransToAdds::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](kevin::AddtestOp op) {
    // llvm::errs() << "Hello: ";
    // llvm::errs().write_escaped(op.getName()) << '\n';
    auto loc = op.getLoc();
    auto operands = op.getOperands();

    OpBuilder b(op.getOperation());
    auto numType = op.values()[0].getType();

    Value add;
    if (numType.isa<IntegerType>()) {
    llvm::errs() << "lowering multi int ,need to fix in rewritepattern in the future\n ";
        add = b.create<AddIOp>(loc, op.getOperand(0), op.getOperand(1));
        for (unsigned i = 2; i < operands.size(); i++) {
          add = b.create<AddIOp>(loc, add, op.getOperand(i));
        }
    } else {
    llvm::errs() << "lowering multi float ,need to fix in rewritepattern in the future\n ";
        add = b.create<AddFOp>(loc, op.getOperand(0), op.getOperand(1));
        for (unsigned i = 2; i < operands.size(); i++) {
          add = b.create<AddFOp>(loc, add, op.getOperand(i));
        }

    }

    op.output().replaceAllUsesWith(add);
    op.erase();
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::kevin::createMultiAddTransPass() {
  return std::make_unique<MultAddTransToAdds>();
}
