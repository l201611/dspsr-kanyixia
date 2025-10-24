/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MultiFile.h"

#include "Error.h"
#include "templates.h"
#include "relative_path.h"
#include "dirutil.h"
#include "strutil.h"

#include <algorithm>
#include <math.h>

using namespace std;

dsp::MultiFile::MultiFile (const char* name) : File (name)
{
}

dsp::MultiFile::~MultiFile ()
{
}

//! Returns true if filename is an ASCII file listing valid filenames
bool dsp::MultiFile::is_valid (const char* metafile) const
{
  if (verbose)
    cerr << "dsp::MultiFile::is_valid meta filename=" << metafile << endl;

  vector<string> tmp_filenames;
  return validate_filenames (tmp_filenames, metafile);
}

bool dsp::MultiFile::validate_filenames (vector<string>& filenames, const char* metafile) const
{
  stringfload (&filenames, metafile);

  if (verbose)
    cerr << "dsp::MultiFile::validate_filenames " << filenames.size() << " filenames read" << endl;

  if (filenames.size() == 0)
    return false;

  string path = dirname (metafile);

  for (unsigned i=0; i < filenames.size(); i++)
  {
    bool found = relative_path (metafile, filenames[i]);
    if (!found)
    {
      if (verbose)
        cerr << "dsp::MultiFile::validate_filenames '" << filenames[i] << "' not found" << endl;
      return false;
    }    

    if (filenames[i] == metafile)
    {
      cerr << "dsp::MultiFile refusing to recursively open meta file (file list lists itself)" << endl;
      return false;
    }
  }

  return true;
}

//! Open the files listed in the ASCII file of filenames
void dsp::MultiFile::open_file (const char* metafile)
{
  if (verbose)
    cerr << "dsp::MultiFile::open_file meta filename=" << metafile << endl;

  vector<string> tmp_filenames;
  validate_filenames (tmp_filenames, metafile);
  open (tmp_filenames);
}

/*!
  \post Resets the file pointers 
*/
void dsp::MultiFile::open (const vector<string>& new_filenames)
{
  if (new_filenames.empty())
    throw Error (InvalidParam, "dsp::Multifile::open",
		 "An empty list of filenames has been given to this method");

  // open up each of the new files and add it to our list of files
  for (unsigned i=0; i<new_filenames.size(); i++)
  {
    string filename = new_filenames[i];

    if (found(filename,filenames))
    {
      cerr << "dsp::MultiFile::open " << filename << " already open" << endl;
      continue;
    }

    File* loader = File::create( filename );
    files.push_back( loader );
    filename = loader->get_filename();
    filenames.push_back( filename );

    after_open(loader);

    if (verbose)
	    cerr << "dsp::MultiFile::open new File = " << filename << endl;
  }

  setup();
}

void dsp::MultiFile::after_open (File*)
{
  // do nothing
}

void dsp::MultiFile::setup ()
{
  // do nothing
}

