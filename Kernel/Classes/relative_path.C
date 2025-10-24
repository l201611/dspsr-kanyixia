/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "relative_path.h"
#include "dirutil.h"
#include "strutil.h"

using namespace std;

// adds the dirname of relative to filename, if it exists at this location, returns false otherwise
bool relative_path (const std::string& relative, std::string& filename)
{
  if (file_exists( filename.c_str() ))
    return true;

  string path = dirname (relative);
  string trial = path + "/" + filename;

  if (!file_exists( trial.c_str() ))
    return false;

  filename = trial;
  return true;
}
