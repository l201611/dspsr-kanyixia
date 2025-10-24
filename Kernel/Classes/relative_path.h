/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
// dspsr/Kernel/Classes/relative_path.h

/*
 * Use the standard C integer types
 */

#ifndef __relative_path_H
#define __relative_path_H

#include <string>

// adds the dirname of relative to filename, if necessary
// returns false if filename is not found
bool relative_path (const std::string& relative, std::string& filename);

#endif /* relative_path_H */
