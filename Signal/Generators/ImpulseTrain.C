/**************************************************************************2
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ImpulseTrain.h"

using namespace std;

dsp::ImpulseTrain::ImpulseTrain () : Generator("ImpulseTrain")
{
  auto info = get_info();
  info->set_nchan(1);
  info->set_npol(1);
  info->set_ndim(2);
  info->set_state(Signal::Analytic);
}

void dsp::ImpulseTrain::operation ()
{
  if (period == 0)
    throw Error (InvalidState, "dsp::ImpulseTrain::operation", "period == 0");

  if (verbose)
    cerr << "dsp::ImpulseTrain::operation block_size=" << block_size << endl;

  output->copy_configuration(get_info());
  output->resize(block_size);
  output->set_input_sample(current_sample);

  if (block_size == 0)
    return;

  if (verbose)
    cerr << "dsp::ImpulseTrain::operation zero output" << endl;

  auto data = output->get_datptr();
  output->zero();

  if (verbose)
    cerr << "dsp::ImpulseTrain::operation initialize impulses" << endl;

  for (uint64_t idat = next - current_sample; idat < block_size; idat += period)
  {
    // the impulses are real-valued (and the imaginary part is left zero)
    data[idat*2] = amplitude;
    next += period;
  }

  current_sample += block_size;

  if (verbose)
    cerr << "dsp::ImpulseTrain::operation done" << endl;
}
