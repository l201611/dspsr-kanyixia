/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Expect.h"
#include "Error.h"
#include <string.h>

using namespace std;

FILE* Expect::fptr() { return file; }

Expect::Expect (const std::string& filename)
{
  file = fopen (filename.c_str(), "r");
  if (!file)
    throw Error (FailedSys, "Expect ctor", "fopen("+filename+")");
}

bool Expect::expect (const std::string& text)
{
  unsigned length = text.length();

  std::vector<char> next (length+1);
  if (fgets (next.data(), length+1, file) == NULL)
    throw Error (InvalidParam, "Expect::expect", "fgets");

  if (text != next.data())
    return false;

  return true;
}
