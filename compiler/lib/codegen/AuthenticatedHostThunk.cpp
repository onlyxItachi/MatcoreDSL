#include "AuthenticatedHostThunk.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include <set>

namespace matcore::mdslc::codegen {
namespace {

// Names are deliberately absent: separate Clang modules have distinct
// identified Type objects. Compare complete by-value structure before and after
// LLVM's normal type remapping. Identified type pointer identity is not an ABI
// property: LLVM can retain distinct isomorphic sret/byval types after linking.
bool sameType(llvm::Type *a, llvm::Type *b, unsigned depth=0) {
  if(depth>64 || a->getTypeID()!=b->getTypeID()) return false;
  if(auto *i=llvm::dyn_cast<llvm::IntegerType>(a))
    return i->getBitWidth()==llvm::cast<llvm::IntegerType>(b)->getBitWidth();
  if(auto *p=llvm::dyn_cast<llvm::PointerType>(a))
    return p->getAddressSpace()==llvm::cast<llvm::PointerType>(b)->getAddressSpace();
  if(auto *s=llvm::dyn_cast<llvm::StructType>(a)) {
    auto *t=llvm::cast<llvm::StructType>(b);
    if(s->isOpaque() || t->isOpaque() || s->isPacked()!=t->isPacked() ||
       s->getNumElements()!=t->getNumElements()) return false;
    for(unsigned i=0;i<s->getNumElements();++i)
      if(!sameType(s->getElementType(i),t->getElementType(i),depth+1)) return false;
    return true;
  }
  if(auto *s=llvm::dyn_cast<llvm::ArrayType>(a)) {
    auto *t=llvm::cast<llvm::ArrayType>(b);
    return s->getNumElements()==t->getNumElements() &&
           sameType(s->getElementType(),t->getElementType(),depth+1);
  }
  if(auto *f=llvm::dyn_cast<llvm::FunctionType>(a)) {
    auto *g=llvm::cast<llvm::FunctionType>(b);
    if(f->isVarArg() || g->isVarArg() || f->getNumParams()!=g->getNumParams() ||
       !sameType(f->getReturnType(),g->getReturnType(),depth+1)) return false;
    for(unsigned i=0;i<f->getNumParams();++i)
      if(!sameType(f->getParamType(i),g->getParamType(i),depth+1)) return false;
    return true;
  }
  // This initial host ABI has ordinary scalar/record arguments, not target
  // vector, token, coroutine, AMX, scalable, or target-extension types.
  return a->isVoidTy() || a->isFloatTy() || a->isDoubleTy();
}

bool sameAttributes(llvm::AttributeSet a,llvm::AttributeSet b) {
  if(a.getNumAttributes()!=b.getNumAttributes()) return false;
  auto j=b.begin();
  for(auto i:a) {
    const auto other=*j++;
    if(i.cmpKind(other)!=0) return false;
    if(i.isTypeAttribute()) {
      if(!other.isTypeAttribute() || !sameType(i.getValueAsType(),other.getValueAsType())) return false;
    } else if(i!=other) return false;
  }
  return true;
}

bool sameAbi(const llvm::Function &a,const llvm::Function &b,bool remapped) {
  // Unlike an sret/byval memory type, the actual LLVM SSA call signature must
  // have identical Type objects after linking: no aggregate value conversion
  // is synthesized by this forwarding thunk.
  if(a.getCallingConv()!=b.getCallingConv() ||
     a.getAddressSpace()!=b.getAddressSpace() ||
     !sameType(a.getFunctionType(),b.getFunctionType()) ||
     (remapped && a.getFunctionType()!=b.getFunctionType()) ||
     !sameAttributes(a.getAttributes().getRetAttrs(),b.getAttributes().getRetAttrs())) return false;
  for(unsigned i=0;i<a.arg_size();++i)
    if(!sameAttributes(a.getAttributes().getParamAttrs(i),b.getAttributes().getParamAttrs(i))) return false;
  for(const auto *name:{"target-cpu","target-features","tune-cpu"})
    if(a.getFnAttribute(name)!=b.getFnAttribute(name)) return false;
  // A forwarding call cannot implement a missing noexcept boundary.
  return !a.doesNotThrow() || b.doesNotThrow();
}

bool supportedFunction(const llvm::Function &function) {
  if(function.isDeclaration() || function.isIntrinsic() ||
     function.isVarArg() || function.hasGC() || function.hasPrefixData() ||
     function.hasPrologueData() || function.hasFnAttribute(llvm::Attribute::Naked) ||
     function.hasFnAttribute(llvm::Attribute::NoReturn) ||
     function.hasFnAttribute(llvm::Attribute::ReturnsTwice) ||
     function.hasFnAttribute(llvm::Attribute::PresplitCoroutine)) return false;
  for(const auto &block:function) if(block.hasAddressTaken()) return false;
  for(unsigned i=0;i<function.arg_size();++i)
    if(function.hasParamAttribute(i,llvm::Attribute::InAlloca) ||
       function.hasParamAttribute(i,llvm::Attribute::Preallocated)) return false;
  return true;
}

llvm::AttributeList withoutBodyFacts(llvm::LLVMContext &context,
                                    llvm::AttributeList attributes,
                                    unsigned arguments,
                                    llvm::AttributeSet function={}) {
  std::vector<llvm::AttributeSet> parameters;
  for(unsigned i=0;i<arguments;++i) parameters.push_back(attributes.getParamAttrs(i));
  return llvm::AttributeList::get(context,function,attributes.getRetAttrs(),parameters);
}

llvm::AttributeSet thunkPolicy(const llvm::Function &host,const llvm::Function &helper) {
  std::vector<llvm::Attribute> result;
  for(auto attr:host.getAttributes().getFnAttrs()) {
    if(attr.isStringAttribute()) {
      const auto name=attr.getKindAsString();
      if(name=="target-cpu" || name=="target-features" || name=="tune-cpu" ||
         name=="frame-pointer" || name=="stack-protector-buffer-size" ||
         name=="probe-stack") result.push_back(attr);
      continue;
    }
    switch(attr.getKindAsEnum()) {
    case llvm::Attribute::UWTable:
    case llvm::Attribute::StackProtect:
    case llvm::Attribute::StackProtectReq:
    case llvm::Attribute::StackProtectStrong:
    case llvm::Attribute::SanitizeAddress:
    case llvm::Attribute::SanitizeThread:
    case llvm::Attribute::SanitizeMemory:
    case llvm::Attribute::SanitizeHWAddress:
    case llvm::Attribute::SanitizeMemTag:
    case llvm::Attribute::NoInline:
    case llvm::Attribute::OptimizeNone:
      result.push_back(attr); break;
    default: break;
    }
  }
  if(helper.doesNotThrow()) result.push_back(llvm::Attribute::get(host.getContext(),llvm::Attribute::NoUnwind));
  return llvm::AttributeSet::get(host.getContext(),result);
}

bool internalUses(const llvm::Value &value,
                  const llvm::SmallPtrSetImpl<const llvm::Function*> &retired,
                  llvm::SmallPtrSetImpl<const llvm::Value*> &seen) {
  if(!seen.insert(&value).second) return true;
  for(const auto *user:value.users()) {
    if(const auto *instruction=llvm::dyn_cast<llvm::Instruction>(user)) {
      if(!retired.contains(instruction->getFunction())) return false;
    } else if(llvm::isa<llvm::ConstantExpr>(user)) {
      if(!internalUses(*user,retired,seen)) return false;
    } else return false; // Includes escaped globals, aliases and block addresses.
  }
  return true;
}

class LinkDiagnostics final:public llvm::DiagnosticHandler {
public:
  explicit LinkDiagnostics(std::string &error):error_(error) {}
  bool handleDiagnostics(const llvm::DiagnosticInfo &info) override {
    llvm::raw_string_ostream stream(error_);
    llvm::DiagnosticPrinterRawOStream printer(stream);
    info.print(printer); stream << '\n'; return true;
  }
private:
  std::string &error_;
};
class DiagnosticScope {
public:
  DiagnosticScope(llvm::LLVMContext &context,std::string &error)
      :context_(context),prior_(context.getDiagnosticHandler()) {
    context_.setDiagnosticHandler(std::make_unique<LinkDiagnostics>(error));
  }
  ~DiagnosticScope() { context_.setDiagnosticHandler(std::move(prior_)); }
private:
  llvm::LLVMContext &context_;
  std::unique_ptr<llvm::DiagnosticHandler> prior_;
};
} // namespace

HostThunkResult linkAuthenticatedHostThunk(const llvm::Module &host,
                                           const llvm::Module &helper,
                                           const HostThunkRequest &request) {
  HostThunkResult result;
  auto reject=[&](std::string error) { result.error=std::move(error); return std::move(result); };
  if(&host.getContext()!=&helper.getContext()) return reject("host/helper LLVM contexts differ");
  if(host.getTargetTriple().empty() || host.getTargetTriple()!=helper.getTargetTriple() ||
     host.getDataLayoutStr().empty() || host.getDataLayoutStr()!=helper.getDataLayoutStr())
    return reject("host/helper target triple or data layout differs");
  if(request.host_symbol.empty() || request.helper_symbol.empty() || request.host_symbol==request.helper_symbol)
    return reject("host/helper symbols must be distinct explicit bindings");
  const auto *original=host.getFunction(request.host_symbol),*implementation=helper.getFunction(request.helper_symbol);
  if(!original || !implementation || !supportedFunction(*original) || !supportedFunction(*implementation))
    return reject("host/helper must be supported concrete function definitions");
  if(host.getNamedValue(request.helper_symbol) || helper.getNamedValue(request.host_symbol))
    return reject("host/helper entry symbol collides across modules");
  std::string verify_error;
  llvm::raw_string_ostream diagnostics(verify_error);
  if(llvm::verifyModule(host,&diagnostics) || llvm::verifyModule(helper,&diagnostics))
    return reject("invalid input LLVM module: "+verify_error);
  if(!sameAbi(*original,*implementation,false)) return reject("host/helper recursive ABI contract differs before linking");
  result.context=std::make_unique<llvm::LLVMContext>();
  auto &context=*result.context;
  auto isolate=[&](const llvm::Module &input) -> std::unique_ptr<llvm::Module> {
    llvm::SmallVector<char,0> bytes;
    llvm::raw_svector_ostream stream(bytes);
    llvm::WriteBitcodeToFile(input,stream);
    auto parsed=llvm::parseBitcodeFile(llvm::MemoryBufferRef(
      llvm::StringRef(bytes.data(),bytes.size()),"authenticated internal module"),context);
    if(!parsed) { result.error=llvm::toString(parsed.takeError()); return {}; }
    return std::move(*parsed);
  };
  auto linked=isolate(host),addition=isolate(helper);
  if(!linked || !addition) return reject("internal module isolation failed: "+result.error);
  // This module is compiler-owned implementation, not a second user TU whose
  // weak definitions should be selected by ordinary host ODR/linker rules.
  // Private inline constructors, cleanup and STL implementation must retain
  // their issued bodies even when the host defines the same linker spelling.
  auto appending=[](const llvm::GlobalValue &global) {
    const auto name=global.getName();
    return global.hasAppendingLinkage() &&
      (name=="llvm.global_ctors" || name=="llvm.global_dtors" ||
       name=="llvm.global.annotations" || name=="llvm.used" || name=="llvm.compiler.used");
  };
  llvm::internalizeModule(*addition,[&](const llvm::GlobalValue &global) {
    return global.getName()==request.helper_symbol || appending(global);
  });
  for(const auto &global:addition->global_values())
    if(!global.isDeclarationForLinker() && !global.hasLocalLinkage() &&
       global.getName()!=request.helper_symbol && !appending(global))
      return reject("compiler helper definition remains externally replaceable: "+global.getName().str());
  if(llvm::verifyModule(*addition,&diagnostics))
    return reject("invalid encapsulated helper LLVM module: "+verify_error);
  bool failed=false;
  {
    DiagnosticScope scope(context,result.error);
    failed=llvm::Linker::linkModules(*linked,std::move(addition));
  }
  if(failed) return reject("LLVM module link failed: "+result.error);
  auto *target=linked->getFunction(request.host_symbol),*callee=linked->getFunction(request.helper_symbol);
  if(!target || !callee || !sameAbi(*target,*callee,true)) {
    std::string diagnostic="host/helper recursive ABI or LLVM call type differs after linking";
    if(target && callee) {
      llvm::raw_string_ostream out(diagnostic);
      out << "\nhost attributes: "; target->getAttributes().print(out);
      out << "\nhelper attributes: "; callee->getAttributes().print(out);
      out << "\nhost function type: "; target->getFunctionType()->print(out);
      out << "\nhelper function type: "; callee->getFunctionType()->print(out);
      out << "\nstructural match: " << sameAbi(*target,*callee,false);
    }
    return reject(diagnostic);
  }
  const auto attributes=withoutBodyFacts(context,target->getAttributes(),target->arg_size(),thunkPolicy(*target,*callee));
  const auto linkage=target->getLinkage();
  auto *debug=target->getSubprogram();
  llvm::SmallVector<std::pair<unsigned,llvm::MDNode*>,4> metadata;
  target->getAllMetadata(metadata);
  // Existing ordinary direct calls can carry the old function's inferred memory
  // facts independently of the definition. Drop those, not their ABI attributes.
  for(auto &function:*linked) for(auto &block:function) for(auto &instruction:block)
    if(auto *call=llvm::dyn_cast<llvm::CallBase>(&instruction))
      if(call->getCalledOperand()->stripPointerCastsAndAliases()==target)
        call->setAttributes(withoutBodyFacts(context,call->getAttributes(),call->arg_size()));
  target->deleteBody(); target->setLinkage(linkage); target->setAttributes(attributes);
  target->setPersonalityFn(nullptr); target->clearMetadata();
  if(debug) target->setSubprogram(debug);
  for(auto [kind,node]:metadata)
    if(kind==llvm::LLVMContext::MD_type || kind==llvm::LLVMContext::MD_kcfi_type)
      target->addMetadata(kind,*node);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context,"mdsl.host.thunk",target));
  if(debug) builder.SetCurrentDebugLocation(llvm::DILocation::get(context,debug->getLine(),0,debug));
  std::vector<llvm::Value*> arguments;
  for(auto &argument:target->args()) arguments.push_back(&argument);
  auto *call=builder.CreateCall(callee,arguments);
  call->setCallingConv(callee->getCallingConv());
  call->setAttributes(withoutBodyFacts(context,callee->getAttributes(),callee->arg_size()));
  if(target->getReturnType()->isVoidTy()) builder.CreateRetVoid(); else builder.CreateRet(call);

  llvm::SmallPtrSet<const llvm::Function*,16> retired;
  std::vector<llvm::Function*> removal;
  for(const auto &name:request.retired_value_functions) {
    auto *function=linked->getFunction(name);
    if(!function || function==target || function==callee || !retired.insert(function).second)
      return reject("retired Value helper binding is missing, duplicated, or an entry");
    removal.push_back(function);
  }
  for(const auto *function:removal) {
    llvm::SmallPtrSet<const llvm::Value*,16> seen;
    if(!internalUses(*function,retired,seen))
      return reject("retired Value helper has a remaining nonhelper use: "+function->getName().str());
  }
  for(auto *function:removal) function->dropAllReferences();
  for(auto *function:removal) function->removeDeadConstantUsers();
  for(auto *function:removal) function->eraseFromParent();
  verify_error.clear();
  if(llvm::verifyModule(*linked,&diagnostics)) return reject("invalid generated ABI thunk: "+verify_error);
  result.module=std::move(linked);
  return result;
}
} // namespace matcore::mdslc::codegen
