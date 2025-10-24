/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DedispersionPipeConfig.h"
#include <cstring>

using namespace std;

//! Add command line options
void dsp::DedispersionPipe::Config::add_options (CommandLine::Menu& menu)
{
  CommandLine::Argument* arg;

  /* ***********************************************************************

  Dispersion removal Options

  *********************************************************************** */

  menu.add ("\n" "Dispersion removal options:");

  arg = menu.add (filterbank, 'F', "<N>[:D]");
  arg->set_help ("create an N-channel filterbank");
  arg->set_long_help
    ("<N> is the number of channels output by the filterank; e.g. -F 256 \n"
     "\n"
     "Reduce the spectral leakage function bandwidth with -F 256:<M> \n"
     "where <M> is the reduction factor."
     "\n"
     "If DM != 0, coherent dedispersion will be performed \n"
     " - after the filterbank with -F 256 or -F 256:<M>\n"
     " - during the filterbank with -F 256:D \n"
     " - before the filterbank with -F 256:B \n" );

  arg = menu.add (inverse_filterbank, "IF", "<N>[:D]");
  arg->set_help( "create inverse filterbank with N output channels");
  arg->set_long_help
    ("<N> is the number of channels output by the inverse filterank; e.g. -IF 256 \n"
     "\n"
     "If DM != 0, coherent dedispersion will be performed \n"
     " - after the inverse filterbank with -F 256:<M>:<O>\n"
     "   where M is the size of the forward FFT and\n"
     "   O is the size of the overlap region\n"
     " - during the inverse filterbank with -F 256:D \n");

  arg = menu.add (do_deripple, "dr");
  arg->set_help( "Apply deripple correction to inverse filterbank");

  string taper_help = "Available tapering functions:\n"
     "\t hanning, welch, bartlett, tukey, top_hat (default: none)\n";

  arg = menu.add (temporal_apodization_type, "t-taper", "name");
  arg->set_help ("name of temporal apodization/tapering/window function");
  arg->set_long_help (taper_help);

  arg = menu.add (spectral_apodization_type, "f-taper", "name");
  arg->set_help ("name of spectral apodization/tapering/window function");
  arg->set_long_help (taper_help);

  arg = menu.add (dispersion_measure, 'D', "dm");
  arg->set_help ("dispersion measure");

  arg = menu.add (interchan_dedispersion, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  arg = menu.add (this, &Config::set_fft_length, 'x', "nfft|minX");
  arg->set_help ("over-ride optimal transform length");
  arg->set_long_help
    ("either specify the desired transform length; e.g, -x 32768 \n"
     "or request the minimum possible length be used via -x min\n"
     "or a multiple of the minimum length; e.g. -x minX2");
     
  arg = menu.add (dynamic_response_filename, "dyn_resp", "file");
  arg->set_help ("load dynamic response from file");

  arg = menu.add (zap_rfi, 'R');
  arg->set_help ("apply time-variable narrow-band RFI filter");

  arg = menu.add (calibrator_database_filename, "pac", "dbase");
  arg->set_help ("calibrator database for phase-coherent matrix convolution");
  arg->set_long_help
    ("specify the name of a database created by pac from which to select\n"
     "the polarization calibrator to be used for matrix convolution");

  arg = menu.add (coherent_derotation, "derotate");
  arg->set_help ("enable phase-coherent Faraday rotation correction");

  arg = menu.add (rotation_measure, "rm");
  arg->set_help ("Faraday rotation measure used for coherent correction");
  
  arg = menu.add (use_fft_bench, "fft-bench");
  arg->set_help ("use benchmark data to choose optimal FFT length");
}

//! Return true if convolution is enabled
bool dsp::DedispersionPipe::Config::convolution_enabled() const
{
  return coherent_dedispersion && get_convolve_when() != Filterbank::Config::During;
}

//! Return true if the (convolving) filterbank is enabled
bool dsp::DedispersionPipe::Config::filterbank_enabled() const
{
  return filterbank.get_nchan() > 1;
}

//! Return true if the inverse filterbank is enabled
bool dsp::DedispersionPipe::Config::inverse_filterbank_enabled() const
{
  return inverse_filterbank.get_nchan() > 0;
}

dsp::Filterbank::Config::When dsp::DedispersionPipe::Config::get_convolve_when() const
{
  if (inverse_filterbank_enabled())
  {
    return inverse_filterbank.get_convolve_when();
  }
  
  if (filterbank_enabled())
  {
    return filterbank.get_convolve_when();
  }

  // by default, Convolution is performed after (or without) any (Inverse) Filterbank
  return Filterbank::Config::After;
}

unsigned dsp::DedispersionPipe::Config::get_filterbank_nchan() const
{
  if (inverse_filterbank.get_nchan() > 1)
  {
    return inverse_filterbank.get_nchan();
  }
  
  if (filterbank.get_nchan() > 1)
  {
    return filterbank.get_nchan();
  }

  return 1;
}

void dsp::DedispersionPipe::Config::set_fft_length(const std::string& fft_length)
{
  char* carg = strdup( fft_length.c_str() );
  char* colon = strchr (carg, ':');
  if (colon)
  {
    *colon = '\0';
    colon++;
    if (sscanf (colon, "%d", &nsmear) < 1)
    {
      throw Error (InvalidParam, "DedispersionPipe::Config::set_fft_length",
                  "error parsing '%s' as filterbank frequency resolution\n", colon);
    }
  }

  unsigned times = 0;

  if (string(carg) == "min")
    times_minimum_nfft = 1;
  else if ( sscanf(carg, "minX%u", &times) == 1 )
    times_minimum_nfft = times;
  else
  {
    unsigned nfft = strtol (carg, 0, 10);
    if (colon && nsmear >= nfft)
    {
      throw Error (InvalidParam, "DedispersionPipe::Config::set_fft_length", 
                  "nfft=%d must be greater than nsmear=%d", nfft, nsmear);
    }
    filterbank.set_freq_res( nfft );
  }
  free(carg);
}
