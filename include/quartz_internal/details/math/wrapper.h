#ifndef MATH_WRAPPER_H
#define MATH_WRAPPER_H

namespace math {

template<typename T>
using ElementaryFunctionType = std::variant<
    T,
    Polynomial < T>,
    Exponential <T>,
    GaussianWithPoly <T>,
    Sinusoidal <T>
>;

enum OperatorType {
  Function,
  Sum,
  Subtract,
  Multiply,
  Divide
};
}

template<typename T>
struct ElementaryFunction {

  math::ElementaryFunctionType<T> value;

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    const auto visit_at = [&position](auto && var) -> auto {
      return quartz::at(var, position);
    };

    return std::visit(visit_at, this->value);
  }

  inline
  ElementaryFunction<T> derivative(const arma::uword index) const {
    const auto visit_derivative = [index](auto && var) -> auto {
      return ElementaryFunction{quartz::derivative(var, index)};
    };

    return std::visit(visit_derivative, this->value);
  }

  ElementaryFunction & operator=(const ElementaryFunction &) = default;
};

template<typename T>
struct MathObject {

private:

  inline
  MathObject(const std::unique_ptr<MathObject> & object) {
    if(object->type == math::OperatorType::Function) {
      this->first = nullptr;
      this->second = nullptr;
      this->value = object->value;
      this->type = object->type;
    } else {
      this->first = std::make_unique<MathObject>(* object->first.get());
      this->second = std::make_unique<MathObject>(* object->second.get());
      this->value = object->value;
      this->type = object->type;
    }
  }

  inline
  MathObject(const std::unique_ptr<MathObject> & first,
             const std::unique_ptr<MathObject> & second,
             const math::OperatorType type) {
    if (type == math::OperatorType::Function) {
      throw Error("MathObject: mismatch between operator type and input");
    }
    this->value = std::nullopt;
    this->first = std::make_unique<MathObject>(* first.get());
    this->second = std::make_unique<MathObject>(* second.get());
    this->type = type;
  }

  std::unique_ptr<MathObject> clone(const MathObject * obj) const {

    std::unique_ptr<MathObject> result = std::make_unique<MathObject>();
    if (obj->type == math::OperatorType::Function) {
      result->value = obj->value;
      result->type = obj->type;
    } else {
      result->first = result->clone(obj->first.get());
      result->second = result->clone(obj->second.get());
      result->type = obj->type;
    }

    return result;
  }

  MathObject clone() const {

    std::unique_ptr<MathObject> result = std::move(this->clone(this));

    return std::move(*result.get());
  }

public:

  std::optional<ElementaryFunction<T>> value;
  std::unique_ptr<MathObject> first;
  std::unique_ptr<MathObject> second;
  math::OperatorType type;


  inline
  MathObject() :
      value({0.0}),
      first(nullptr),
      second(nullptr),
      type(math::OperatorType::Function) {}


  inline
  MathObject(const ElementaryFunction<T> & function) :
      value(function),
      first(nullptr),
      second(nullptr),
      type(math::OperatorType::Function) {}

  inline
  MathObject(const MathObject & object) {
    std::unique_ptr<MathObject> result = std::move(this->clone(&object));
    this->first = std::move(result->first);
    this->second = std::move(result->second);
    this->value = std::move(result->value);
    this->type = std::move(result->type);
  }


  inline
  MathObject(const MathObject & first,
             const MathObject & second,
             const math::OperatorType type) {
    this->value = std::nullopt;
    this->first = std::make_unique<MathObject>(first);
    this->second = std::make_unique<MathObject>(second);
    this->type = type;
  }

  template<typename U>
  std::common_type_t<T, U> at(const arma::Col<U> & position) const {
    if (this->type == math::OperatorType::Function) {
      return this->value->at(position);
    } else if (this->type == math::OperatorType::Sum) {
      return this->first->at(position) +
             this->second->at(position);
    } else if (this->type == math::OperatorType::Subtract) {
      return this->first->at(position) -
             this->second->at(position);
    } else if (this->type == math::OperatorType::Multiply) {
      return this->first->at(position) *
             this->second->at(position);
    } else if (this->type == math::OperatorType::Divide) {
      return this->first->at(position) /
             this->second->at(position);
    } else {
      throw Error("MathObject: Unknown error for at");
    }

  }

  inline
  MathObject derivative(const arma::uword index) const {
    if(this->type == math::OperatorType::Function) {
      const ElementaryFunction<T> result = this->value->derivative(index);
      return MathObject(result);
    } else if(this->type == math::OperatorType::Sum) {
      return this->first->derivative(index) + this->second->derivative(index);
    } else if(this->type == math::OperatorType::Subtract) {
      return this->first->derivative(index) - this->second->derivative(index);
    } else if(this->type == math::OperatorType::Multiply) {
      return (this->first->derivative(index) * MathObject(this->second)) +
      (this->second->derivative(index) * MathObject(this->first));
    } else if(this->type == math::OperatorType::Divide) {
      return (this->first->derivative(index) * MathObject(this->second) -
              this->second->derivative(index) * MathObject(this->first)) /
          (MathObject(this->second, this->second, math::OperatorType::Multiply));
    } else {
      throw Error("Unknown error for derivative");
    }
  }

  inline
  arma::uword dim() const {
    const auto visit_dim = [](auto && var) -> arma::uword {
      return var.dim();
    };

    return std::visit(visit_dim, this->value);
  }

    MathObject operator + (const MathObject & obj) const {
      return MathObject(*this, obj, math::OperatorType::Sum);
    }

    MathObject operator - (const MathObject & obj) const {
      return MathObject(*this, obj, math::OperatorType::Subtract);
    }

    MathObject operator * (const MathObject & obj) const {
      return MathObject(*this, obj, math::OperatorType::Multiply);
    }

    MathObject operator / (const MathObject & obj) const {
      return MathObject(*this, obj, math::OperatorType::Divide);
    }

  MathObject operator + (const ElementaryFunction<T> & obj) const {
    return MathObject(*this, MathObject(obj), math::OperatorType::Sum);
  }

  MathObject operator - (const ElementaryFunction<T> & obj) const {
    return MathObject(*this, MathObject(obj), math::OperatorType::Subtract);
  }

  MathObject operator * (const ElementaryFunction<T> & obj) const {
    return MathObject(*this, MathObject(obj), math::OperatorType::Multiply);
  }

  MathObject operator / (const ElementaryFunction<T> & obj) const {
    return MathObject(*this, MathObject(obj), math::OperatorType::Divide);
  }

  MathObject & operator=(const MathObject & obj) {
    return * clone(& obj).get();
  }

  MathObject & operator=(MathObject && other) = default;
};


#endif //MATH_WRAPPER_H
