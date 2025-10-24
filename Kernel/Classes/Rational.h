//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Rational_h
#define __Rational_h

#include <iostream>

//! Represents a rational number
class Rational
{
  friend std::istream& operator >> (std::istream& in, Rational& r);
  friend std::ostream& operator << (std::ostream& out, const Rational& r);

public:

  explicit Rational(int numerator = 0, unsigned denominator = 1);

  const Rational& operator = (const Rational&);
  bool operator == (const Rational&) const;
  bool operator != (const Rational&) const;

  const Rational& operator = (int num);
  bool operator == (int num) const;
  bool operator != (int num) const;

  int operator * (int num) const;

  double doubleValue( ) const;

  /* divides argument by self and throws an exception if the
   result is not an integer */
  int normalize (int) const ;

  // double normalize (double) const;

  int get_numerator () const {
    return numerator;
  }

  unsigned get_denominator () const {
    return denominator;
  }

private:
  int numerator;
  unsigned denominator;
  void reduce( );

};

template<typename T>
T operator * (T x, const Rational& q)
{
  return (x * q.get_numerator()) / q.get_denominator();
}

template<typename T>
T operator / (T x, const Rational& q)
{
  return (x * q.get_denominator()) / q.get_numerator();
}

template<typename T>
bool operator > (const Rational& q, T x)
{
  return q.get_numerator() > (x * q.get_denominator());
}

template<typename T>
bool operator < (const Rational& q, T x)
{
  return q.get_numerator() < (x * q.get_denominator());
}

#endif // !defined(__Rational_h)
