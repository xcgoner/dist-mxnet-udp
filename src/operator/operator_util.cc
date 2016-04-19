/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator_util.cc
 *  Implementation of operator util.
 */
#include <mxnet/operator_util.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <mxnet/engine.h>
#include <vector>
#include <mutex>
#include "./operator_common.h"

namespace mxnet {
namespace op {

class SimpleOpPropBase;
class SimpleUnaryOpProp;
class SimpleBinaryOpProp;

class SimpleOpRegEntryImpl : public SimpleOpRegEntry {
 public:
  TSelf& set_symbol_op_name(const std::string& symbol_name) override {
    CHECK(op_reg_ == nullptr || symbol_name == symbol_name_)
        << "need to call set_symbol_op_name before all other calls";
    symbol_name_ = symbol_name;
    return *this;
  }

  TSelf& set_enable_scalar(
      bool enable_scalar,
      SimpleOpScalarOption type_mask) override {
    std::lock_guard<std::mutex> lock(mutex_);
    enable_scalar_ = enable_scalar;
    scalar_type_mask_ = type_mask;
    CHECK(!enable_kwargs_ || !enable_scalar_)
        << "Cannot have both kwargs and scalar arguments";
    return *this;
  }

  TSelf& set_enable_kwargs(bool enable_kwargs) override {
    std::lock_guard<std::mutex> lock(mutex_);
    enable_kwargs_ = enable_kwargs;
    CHECK(!enable_kwargs_ || !enable_scalar_)
        << "Cannot have both kwargs and scalar arguments";
    return *this;
  }

  TSelf& set_shape_function(UnaryShapeFunction fshapeinfer) override {
    std::lock_guard<std::mutex> lock(mutex_);
    unary_shape_ = fshapeinfer;
    return *this;
  }

  TSelf& set_shape_function(BinaryShapeFunction fshapeinfer) override {
    std::lock_guard<std::mutex> lock(mutex_);
    binary_shape_ = fshapeinfer;
    return *this;
  }

  TSelf& set_function(int dev_mask,
                      UnaryFunction funary,
                      SimpleOpInplaceOption inplace_in_out,
                      SimpleOpRegOption register_symbolic) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&funary_, dev_mask, funary, "UnaryFunction");
    unary_forward_inplace_in_out_ = (inplace_in_out == kInplaceInOut);
    if (++reg_counter_ == 1) {
      this->RegisterUnaryImperative();
      register_symbolic_ = (register_symbolic == kRegisterSymbolic);
      if (register_symbolic_) {
        this->RegisterUnarySymbolic();
      }
    }
    return *this;
  }

  TSelf& set_function(int dev_mask,
                      BinaryFunction fbinary,
                      SimpleOpInplaceOption inplace_lhs_out,
                      SimpleOpRegOption register_symbolic) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&fbinary_, dev_mask, fbinary, "BinaryFunction");
    binary_forward_inplace_lhs_out_ = (inplace_lhs_out == kInplaceLhsOut);
    if (++reg_counter_ == 1) {
      this->RegisterBinaryImperative();
      register_symbolic_ = (register_symbolic == kRegisterSymbolic);
      if (register_symbolic_) {
        this->RegisterBinarySymbolic();
      }
    }
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradFunctionT0 fgrad,
                      SimpleOpInplaceOption inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&funary_grad_t0_, dev_mask, fgrad, "UnaryGradFunctionT0");
    unary_backward_inplace_out_in_ = (inplace_out_in_grad == kInplaceOutIn);
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradFunctionT1 fgrad,
                      SimpleOpInplaceOption inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&funary_grad_t1_, dev_mask, fgrad, "UnaryGradFunctionT1");
    unary_backward_inplace_out_in_ = (inplace_out_in_grad == kInplaceOutIn);
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      UnaryGradFunctionT2 fgrad,
                      SimpleOpInplaceOption inplace_out_in_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&funary_grad_t2_, dev_mask, fgrad, "UnaryGradFunctionT2");
    unary_backward_inplace_out_in_ = (inplace_out_in_grad == kInplaceOutIn);
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      BinaryGradFunctionT0 fgrad,
                      SimpleOpInplaceOption inplace_out_lhs_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&fbinary_grad_t0_, dev_mask, fgrad, "BinaryGradFunctionT0");
    binary_backward_inplace_out_lhs_ = (inplace_out_lhs_grad == kInplaceLhsOut);
    return *this;
  }

  TSelf& set_gradient(int dev_mask,
                      BinaryGradFunctionT1 fgrad,
                      SimpleOpInplaceOption inplace_out_lhs_grad) override {
    std::lock_guard<std::mutex> lock(mutex_);
    SetFunction(&fbinary_grad_t1_, dev_mask, fgrad, "BinaryGradFunctionT1");
    binary_backward_inplace_out_lhs_ = (inplace_out_lhs_grad == kInplaceLhsOut);
    return *this;
  }

  TSelf& describe(const std::string &description) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reg_counter_ != 1) return *this;
    NDArrayReg().describe(description);
    if (register_symbolic_) {
      OpReg().describe(description);
    }
    return *this;
  }

  TSelf& add_arguments(const std::vector<dmlc::ParamFieldInfo> &args) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reg_counter_ != 1) return *this;
    NDArrayReg().add_arguments(args);
    if (register_symbolic_) {
      OpReg().add_arguments(args);
    }
    return *this;
  }

 protected:
  // make friend with unary op
  friend class SimpleOpPropBase;
  friend class SimpleUnaryOpProp;
  friend class SimpleBinaryOpProp;
  // internal mutex
  std::mutex mutex_;
  // registration counter
  int reg_counter_{0};
  // whether register symbolic function.
  bool register_symbolic_{true};
  // name of symbolic operator.
  std::string symbol_name_;
  // number of scalar arguments
  bool enable_scalar_{false};
  // type mask of scalar arguments in imperative API.
  SimpleOpScalarOption scalar_type_mask_{kArrayBeforeScalar};
  // whether kwargs is enabled in the function.
  bool enable_kwargs_{false};
  // ------ unary functions -----
  // unary shape inference information.
  UnaryShapeFunction unary_shape_{nullptr};
  // unary functions on each device mask
  std::vector<UnaryFunction> funary_;
  // type 1 gradient function
  std::vector<UnaryGradFunctionT0> funary_grad_t0_;
  // type 2 gradient function
  std::vector<UnaryGradFunctionT1> funary_grad_t1_;
  // type 2 gradient function
  std::vector<UnaryGradFunctionT2> funary_grad_t2_;
  // whether do inplace optimization of in 0 and output
  bool unary_forward_inplace_in_out_{false};
  // whether do inplace optimization of out_grad and in_grad0
  bool unary_backward_inplace_out_in_{false};
  // ------ binary functions -----
  // binary shape inference information.
  BinaryShapeFunction binary_shape_{nullptr};
  // unary functions on each device mask
  std::vector<BinaryFunction> fbinary_;
  // type 1 gradient function
  std::vector<BinaryGradFunctionT0> fbinary_grad_t0_;
  // type 2 gradient function
  std::vector<BinaryGradFunctionT1> fbinary_grad_t1_;
  // whether do inplace optimization of in 0 and output
  bool binary_forward_inplace_lhs_out_{false};
  // whether do inplace optimization of out_grad and in_grad0
  bool binary_backward_inplace_out_lhs_{false};

  template<typename TFunction>
  inline void SetFunction(std::vector<TFunction>* vfunc,
                          int dev_mask,
                          TFunction func,
                          const char* type) {
    if (vfunc->size() <= static_cast<size_t>(dev_mask)) {
      vfunc->resize(dev_mask + 1, nullptr);
    }
    if (vfunc->at(dev_mask) != nullptr) {
      LOG(FATAL) << "Device " << type << " function " << this->name
                 << " already registerd for device " << dev_mask;
    }
    vfunc->at(dev_mask) = func;
  }

 private:
  // internal reference to NDArray registry
  NDArrayFunctionReg* ndarray_reg_{nullptr};
  // internal reference to operator registry
  OperatorPropertyReg* op_reg_{nullptr};
  // internal function to register NDArray function.
  inline NDArrayFunctionReg &NDArrayReg() {
    if (ndarray_reg_ == nullptr) {
      NDArrayFunctionReg &reg =
          ::dmlc::Registry<NDArrayFunctionReg>::Get()->__REGISTER__(this->name);
      ndarray_reg_ = &reg;
    }
    return *ndarray_reg_;
  }
  // internal function to register NDArray function.
  inline OperatorPropertyReg &OpReg() {
    if (op_reg_ == nullptr) {
      if (symbol_name_.length() == 0) {
        symbol_name_ = this->name;
      }
      OperatorPropertyReg &reg =
          ::dmlc::Registry<OperatorPropertyReg>::Get()->__REGISTER__(symbol_name_);
      op_reg_ = &reg;
    }
    return *op_reg_;
  }
  // register unary function.
  void RegisterUnaryImperative();
  // register unary symbolic function.
  void RegisterUnarySymbolic();
  // register unary function.
  void RegisterBinaryImperative();
  // register unary symbolic function.
  void RegisterBinarySymbolic();
};

SimpleOpRegEntry& SimpleOpRegistry::__REGISTER_OR_FIND__(const std::string &name) {
  if (fmap_.count(name) != 0) return *fmap_.at(name);
  SimpleOpRegEntry *e = new SimpleOpRegEntryImpl();
  e->name = name;
  fmap_[name] = e;
  return *e;
}

SimpleOpRegistry* SimpleOpRegistry::Get() {
  static SimpleOpRegistry inst;
  return &inst;
}

SimpleOpRegistry::~SimpleOpRegistry() {
  for (auto kv : fmap_) {
    delete kv.second;
  }
}
//-------------------------------------
// unary function Implementation
//-------------------------------------
void SimpleOpRegEntryImpl::RegisterUnaryImperative() {
  CHECK_EQ(reg_counter_, 1);
  // The body to be registered
  auto body = [this] (NDArray** used_vars,
                      real_t* s,
                      NDArray** mutate_vars,
                      int num_params,
                      char** param_keys,
                      char** param_vals) {
    NDArray& src = *used_vars[0];
    NDArray* out = mutate_vars[0];
    // setup env.
    EnvArguments env;
    if (enable_scalar_) env.scalar = s[0];
    if (enable_kwargs_) {
      for (int i = 0; i < num_params; ++i) {
        env.kwargs.emplace_back(std::make_pair(
            std::string(param_keys[i]), std::string(param_vals[i])));
      }
    } else {
      CHECK_EQ(num_params, 0)
        << "operator " << this->name << " do not take keyword arguments";
    }
    // shape inference.
    TShape dshape;
    if (unary_shape_ != nullptr) {
      dshape = unary_shape_(src.shape(), env);
    } else {
      dshape = src.shape();
    }
    // check output shape.
    if (out->is_none()) {
      *out = NDArray(dshape, src.ctx(), true, src.dtype());
    } else {
      CHECK(out->ctx() == src.ctx()) << "target context mismatch";
      CHECK(out->dtype() == src.dtype()) << "target data type mismatch";
      CHECK(out->shape() == dshape) << "target shape mismatch "
      << out->shape() << " vs. " << dshape;
    }
    // important: callback must always capture by value
    NDArray ret = *out;
    // get the const variables
    std::vector<Engine::VarHandle> const_vars;
    if (src.var() != ret.var()) {
      const_vars.push_back(src.var());
    }
    // check if the function exist
    int dev_mask = src.ctx().dev_mask();
    // error message
    if (static_cast<size_t>(dev_mask) >= funary_.size() ||
        funary_[dev_mask] == nullptr) {
      if (dev_mask == gpu::kDevMask) {
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      }
      LOG(FATAL) << "Function " << this->name
                 << "not registered for device " << dev_mask;
    }
    // invoke the function
    UnaryFunction fun = funary_[dev_mask];
    OpReqType req = kWriteTo;
    if (src.var() == ret.var()) {
      req = kWriteInplace;
      CHECK(unary_forward_inplace_in_out_)
          << "inplace operation is not enabled for operator " << name;
    }

    Engine::Get()->PushSync([src, ret, fun, dev_mask, req, env](RunContext ctx) {
        ret.CheckAndAlloc();
        TBlob tmp = ret.data();
        (*fun)(src.data(), env, &tmp, req, ctx);
#if MXNET_USE_CUDA
        if (dev_mask == gpu::kDevMask) {
          ctx.get_stream<gpu>()->Wait();
        }
#endif
      }, src.ctx(), const_vars, {ret.var()});
  };
  // register the function.
  NDArrayReg()
      .set_body(body)
      .set_num_use_vars(1)
      .set_num_mutate_vars(1);
  if (enable_scalar_) {
    if (scalar_type_mask_ == kArrayBeforeScalar) {
      NDArrayReg()
          .set_num_scalars(1)
          .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
          .add_argument("src", "NDArray", "Source input to the function")
          .add_argument("scalar", "float", "scalar input to the function");
    } else {
      NDArrayReg()
          .set_num_scalars(1)
          .set_type_mask(kScalarArgBeforeNDArray | kAcceptEmptyMutateTarget)
          .add_argument("scalar", "float", "scalar input to the function")
          .add_argument("src", "NDArray", "Source input to the function");
    }
  } else {
    NDArrayReg()
      .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
      .add_argument("src", "NDArray", "Source input to the function");
  }
}

// operator to invoke unary function.
struct SimpleUnaryOperator : public Operator {
  EnvArguments env;
  UnaryFunction forward;
  UnaryGradFunctionT0 backward0{nullptr};
  UnaryGradFunctionT1 backward1{nullptr};
  UnaryGradFunctionT2 backward2{nullptr};

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    TBlob out = out_data[0];
    (*forward)(in_data[0], env, &out, req[0], ctx.run_ctx);
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad,
                const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    OutputGrad ograd; ograd.data = out_grad[0];
    TBlob igrad = in_grad[0];

    if (backward0 != nullptr) {
      (*backward0)(ograd, env, &igrad, req[0], ctx.run_ctx);
    } else if (backward1 != nullptr) {
      OutputValue out_value; out_value.data = out_data[0];
      (*backward1)(ograd, out_value, env, &igrad, req[0], ctx.run_ctx);
    } else if (backward2 != nullptr) {
      Input0 in0; in0.data = in_data[0];
      (*backward2)(ograd, in0, env, &igrad, req[0], ctx.run_ctx);
    } else {
      LOG(FATAL) << "Backward is not supported";
    }
  }
};  // class SimpleUnaryOperator

struct SimpleOpScalarParam :
      public dmlc::Parameter<SimpleOpScalarParam> {
  float scalar;
  DMLC_DECLARE_PARAMETER(SimpleOpScalarParam) {
    DMLC_DECLARE_FIELD(scalar)
        .describe("scalar value.");
  }
};

DMLC_REGISTER_PARAMETER(SimpleOpScalarParam);

class SimpleOpPropBase : public OperatorProperty {
 public:
  std::string name;
  EnvArguments env;
  SimpleOpRegEntryImpl* source;

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    if (source->enable_kwargs_) {
      env.kwargs = kwargs;
    } else if (source->enable_scalar_) {
      SimpleOpScalarParam param;
      param.Init(kwargs);
      env.scalar = param.scalar;
    } else {
      CHECK_EQ(kwargs.size(), 0)
          << "Operator " << source->symbol_name_ << " donot accept any keyword arguments";
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    if (source->enable_kwargs_) {
      return std::map<std::string, std::string>(
          env.kwargs.begin(), env.kwargs.end());
    } else if (source->enable_scalar_) {
      SimpleOpScalarParam param;
      param.scalar = env.scalar;
      return param.__DICT__();
    } else {
      return std::map<std::string, std::string>();
    }
  }

  std::string TypeString() const override {
    return name;
  }
};

class SimpleUnaryOpProp : public SimpleOpPropBase {
 public:
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    if (source->unary_shape_ == nullptr) {
      out_shape->push_back(dshape);
    } else {
      out_shape->push_back((*(source->unary_shape_))(dshape, env));
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SimpleUnaryOpProp();
    ptr->source = source;
    ptr->name = name;
    ptr->env = env;
    return ptr;
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (source->funary_grad_t0_.size() != 0) {
      return {out_grad[0]};
    } else if (source->funary_grad_t1_.size() != 0) {
      return {out_grad[0], out_data[0]};
    } else if (source->funary_grad_t2_.size() != 0) {
      return {out_grad[0], in_data[0]};
    } else {
      LOG(FATAL) << "Backward of " << name << " is not decalred";
      return {};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    if (source->unary_backward_inplace_out_in_) {
      return {{out_grad[0], in_grad[0]}};
    } else {
      return {};
    }
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (source->unary_forward_inplace_in_out_) {
      return {{in_data[0], out_data[0]}};
    } else {
      return {};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    size_t dev_mask = ctx.dev_mask();
    SimpleUnaryOperator *op = new SimpleUnaryOperator();
    CHECK(dev_mask < source->funary_.size() && source->funary_[dev_mask] != nullptr);
    op->forward = source->funary_[dev_mask];
    op->env = this->env;
    if (dev_mask < source->funary_grad_t0_.size()) {
      op->backward0 = source->funary_grad_t0_[dev_mask];
    }
    if (dev_mask < source->funary_grad_t1_.size()) {
      op->backward1 = source->funary_grad_t1_[dev_mask];
    }
    if (dev_mask < source->funary_grad_t2_.size()) {
      op->backward2 = source->funary_grad_t2_[dev_mask];
    }
    return op;
  }
};

void SimpleOpRegEntryImpl::RegisterUnarySymbolic() {
  // register the operator
  auto op_factory = [this]() {
    SimpleUnaryOpProp *prop = new SimpleUnaryOpProp();
    prop->name = this->symbol_name_;
    prop->source = this;
    return prop;
  };
  OpReg()
      .set_body(op_factory)
      .add_argument("lhs", "Symbol", "Left symbolic input to the function")
      .add_argument("rhs", "Symbol", "Left symbolic input to the function");
}


//-------------------------------------
// binary function Implementation
//-------------------------------------
void SimpleOpRegEntryImpl::RegisterBinaryImperative() {
  CHECK_EQ(reg_counter_, 1);
  // The body to be registered
  auto body = [this] (NDArray** used_vars,
                      real_t* s,
                      NDArray** mutate_vars,
                      int num_params,
                      char** param_keys,
                      char** param_vals) {
    NDArray& lhs = *used_vars[0];
    NDArray& rhs = *used_vars[1];
    NDArray* out = mutate_vars[0];
    // setup env.
    EnvArguments env;
    if (enable_scalar_) env.scalar = s[0];
    if (enable_kwargs_) {
      for (int i = 0; i < num_params; ++i) {
        env.kwargs.emplace_back(std::make_pair(
            std::string(param_keys[i]), std::string(param_vals[i])));
      }
    } else {
      CHECK_EQ(num_params, 0)
        << "operator " << this->name << " do not take keyword arguments";
    }
    // shape inference.
    TShape dshape;
    if (binary_shape_ != nullptr) {
      dshape = binary_shape_(lhs.shape(), rhs.shape(), env);
    } else {
      CHECK_EQ(lhs.shape(), rhs.shape()) << "operands shape mismatch";
      dshape = lhs.shape();
    }
    CHECK_EQ(lhs.ctx(), rhs.ctx())
      << "operands context mismatch " << lhs.ctx().dev_type << " " << lhs.ctx().dev_id << \
      " vs. " << rhs.ctx().dev_type << " " << rhs.ctx().dev_id;
    CHECK_EQ(lhs.dtype(), rhs.dtype()) << "operands type mismatch";

    // check output shape.
    if (out->is_none()) {
      *out = NDArray(dshape, lhs.ctx(), true, lhs.dtype());
    } else {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
      CHECK(out->dtype() == lhs.dtype()) << "target data type mismatch";
      CHECK(out->shape() == dshape) << "target shape mismatch "
      << out->shape() << " vs. " << dshape;
    }
    // important: callback must always capture by value
    NDArray ret = *out;
    // get the const variables
    std::vector<Engine::VarHandle> const_vars;
    if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
    if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

    // check if the function exist
    int dev_mask = lhs.ctx().dev_mask();
    // error message
    if (static_cast<size_t>(dev_mask) >= fbinary_.size() ||
        fbinary_[dev_mask] == nullptr) {
      if (dev_mask == gpu::kDevMask) {
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      }
      LOG(FATAL) << "Function " << this->name
                 << "not registered for device " << dev_mask;
    }
    // invoke the function
    BinaryFunction fun = fbinary_[dev_mask];
    OpReqType req = kWriteTo;
    if (lhs.var() == ret.var()) {
      req = kWriteInplace;
      CHECK(binary_forward_inplace_lhs_out_)
          << "inplace operation is not enabled for operator " << name;
    }
    if (rhs.var() == ret.var()) {
      LOG(ERROR) << " operation " << this->name
        << " warning, perform inplace operation with right operand, may not be supported";
    }

    Engine::Get()->PushSync([lhs, rhs, ret, fun, dev_mask, req, env](RunContext ctx) {
        ret.CheckAndAlloc();
        TBlob tmp = ret.data();
        (*fun)(lhs.data(), rhs.data(), env, &tmp, req, ctx);
#if MXNET_USE_CUDA
        if (dev_mask == gpu::kDevMask) {
          ctx.get_stream<gpu>()->Wait();
        }
#endif
      }, lhs.ctx(), const_vars, {ret.var()});
  };
  // register the function.
  NDArrayReg()
      .set_body(body)
      .set_num_use_vars(2)
      .set_num_mutate_vars(1);
  if (enable_scalar_) {
    if (scalar_type_mask_ == kArrayBeforeScalar) {
      NDArrayReg()
          .set_num_scalars(1)
          .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
          .add_argument("lhs", "NDArray", "Left operand  to the function")
          .add_argument("rhs", "NDArray", "Right operand to the function")
          .add_argument("scalar", "float", "scalar input to the function");
    } else {
      NDArrayReg()
          .set_num_scalars(1)
          .set_type_mask(kScalarArgBeforeNDArray | kAcceptEmptyMutateTarget)
          .add_argument("scalar", "float", "scalar input to the function")
          .add_argument("src", "NDArray", "Source input to the function")
          .add_argument("lhs", "NDArray", "Left operand  to the function")
          .add_argument("rhs", "NDArray", "Right operand to the function");
    }
  } else {
    NDArrayReg()
        .set_type_mask(kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget)
        .add_argument("lhs", "NDArray", "Left operand  to the function")
        .add_argument("rhs", "NDArray", "Right operand to the function");
  }
}


struct SimpleBinaryOperator : public Operator {
  EnvArguments env;
  BinaryFunction forward;
  BinaryGradFunctionT0 backward0{nullptr};
  BinaryGradFunctionT1 backward1{nullptr};

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    TBlob out = out_data[0];
    (*forward)(in_data[0], in_data[1], env, &out, req[0], ctx.run_ctx);
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad,
                const std::vector<TBlob> &aux_args) override {
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    CHECK_EQ(req.size(), 2);
    OutputGrad ograd; ograd.data = out_grad[0];
    TBlob lgrad = in_grad[0];
    TBlob rgrad = in_grad[1];

    if (backward0 != nullptr) {
      (*backward0)(ograd, env,
                   &lgrad, &rgrad, req[0], req[1], ctx.run_ctx);
    } else if (backward1 != nullptr) {
      Input0 in0; in0.data = in_data[0];
      Input1 in1; in1.data = in_data[1];
      (*backward1)(ograd, in0, in1, env,
                   &lgrad, &rgrad, req[0], req[1], ctx.run_ctx);
    } else {
      LOG(FATAL) << "Backward is not supported";
    }
  }
};  // class SimpleBinaryOperator

class SimpleBinaryOpProp : public SimpleOpPropBase {
 public:
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[lhs, rhs]";
    const TShape& lshape = in_shape->at(0);
    const TShape& rshape = in_shape->at(1);
    out_shape->clear();
    if (source->binary_shape_ == nullptr) {
      if (in_shape->at(0).ndim() != 0) {
        SHAPE_ASSIGN_CHECK(*in_shape, 1, in_shape->at(0));
      } else if (in_shape->at(1).ndim() != 0) {
        in_shape->at(0) = in_shape->at(1);
      } else {
        return false;
      }
      out_shape->push_back(lshape);
    } else {
      if (lshape.ndim() == 0) return false;
      if (rshape.ndim() == 0) return false;
      out_shape->push_back((*(source->binary_shape_))(lshape, rshape, env));
    }
    return true;
  }

  std::vector<std::string> ListArguments() const override {
    return {"lhs", "rhs"};
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SimpleBinaryOpProp();
    ptr->source = source;
    ptr->name = name;
    ptr->env = env;
    return ptr;
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (source->fbinary_grad_t0_.size() != 0) {
      return {out_grad[0]};
    } else if (source->fbinary_grad_t1_.size() != 0) {
      return {out_grad[0], in_data[0], in_data[1]};
    } else {
      LOG(FATAL) << "Backward of " << name << " is not decalred";
      return {};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    if (source->binary_backward_inplace_out_lhs_) {
      return {{out_grad[0], in_grad[0]}};
    } else {
      return {};
    }
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    if (source->binary_forward_inplace_lhs_out_) {
      return {{in_data[0], out_data[0]}};
    } else {
      return {};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    size_t dev_mask = ctx.dev_mask();
    SimpleBinaryOperator *op = new SimpleBinaryOperator();
    CHECK(dev_mask < source->fbinary_.size() && source->fbinary_[dev_mask] != nullptr);
    op->forward = source->fbinary_[dev_mask];
    op->env = this->env;
    if (dev_mask < source->fbinary_grad_t0_.size()) {
      op->backward0 = source->fbinary_grad_t0_[dev_mask];
    }
    if (dev_mask < source->fbinary_grad_t1_.size()) {
      op->backward1 = source->fbinary_grad_t1_[dev_mask];
    }
    return op;
  }
};

void SimpleOpRegEntryImpl::RegisterBinarySymbolic() {
  // register the operator
  auto op_factory = [this]() {
    SimpleBinaryOpProp *prop = new SimpleBinaryOpProp();
    prop->name = symbol_name_;
    prop->source = this;
    return prop;
  };
  OpReg()
      .set_body(op_factory)
      .add_argument("lhs", "Symbol", "Left symbolic input to the function")
      .add_argument("rhs", "Symbol", "Left symbolic input to the function");
}

}  // namespace op
}  // namespace mxnet