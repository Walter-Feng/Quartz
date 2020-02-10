#ifndef QUARTZ_CHECK_MEMBER_H
#define QUARTZ_CHECK_MEMBER_H

#include <type_traits>

template<typename, typename T>
struct has_time_evolve {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_time_evolve<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().time_evolve( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};



template<typename, typename T>
struct has_derivative {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_derivative<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().derivative( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};




template<typename, typename T>
struct has_at {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_at<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().at( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};



template<typename, typename T>
struct has_propagation_type {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_propagation_type<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().propagation_type( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};




template<typename, typename T>
struct has_inv {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_inv<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().inv( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};



template<typename, typename T>
struct has_positional_expectation {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_positional_expectation<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().positional_expectation( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};




template<typename, typename T>
struct has_momentum_expectation {
  static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_momentum_expectation<C, Ret(Args...)> {
private:
  template<typename T>
  static constexpr auto check(T*)
  -> typename
  std::is_same<
      decltype( std::declval<T>().momentum_expectation( std::declval<Args>()... ) ),
      Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  >::type {
    return nullptr;
  }
  // attempt to call it and see if the return type is correct

  template<typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<C>(0)) type;

public:
  static constexpr bool value = type::value;
};




#endif //QUARTZ_CHECK_MEMBER_H
