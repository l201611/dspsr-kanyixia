/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ExamplePipeline.h"

#include "dsp/Source.h"
#include "dsp/Convolution.h"
#include "dsp/SampleDelay.h"
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Detection.h"

using namespace std;

void dsp::ExamplePipeline::construct ()
{
  Reference::To<TimeSeries> timeseries = get_source()->get_output();

  Reference::To<Response> kernel = new Dedispersion;

  Reference::To<Convolution> convolution = new Convolution;
  convolution->set_input( timeseries );
  convolution->set_output( timeseries = new_TimeSeries() );
  convolution->set_response( kernel );
  append(convolution);

  Reference::To<SampleDelay> delay = new SampleDelay;
  delay->set_input (timeseries);
  delay->set_output (timeseries = new_TimeSeries() );
  delay->set_function (new Dedispersion::SampleDelay);
  append( delay );

  Reference::To<Detection> detection = new Detection;
  detection->set_input( timeseries );
  detection->set_output( timeseries = new_TimeSeries() );
  append( detection );
}

void dsp::ExamplePipeline::prepare ()
{
}

