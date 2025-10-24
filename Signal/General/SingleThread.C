/***************************************************************************
 *
 *   Copyright (C) 2011-2025 by Willem van Straten, Will Gauvin,
 *   and Jesmigel Cantos
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "environ.h"
#include "dsp/SingleThread.h"
#include "dsp/SourceOpen.h"

#include "dsp/IOManager.h"
#include "dsp/Seekable.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "dsp/Scratch.h"
#include "dsp/MemoryHost.h"

#include "dsp/WeightedTimeSeries.h"

#include "RealTimer.h"

#if HAVE_CFITSIO
#if HAVE_fits
#include "dsp/FITSUnpacker.h"
#endif
#endif

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#endif

#include "dsp/ObservationChange.h"
#include "dsp/Dump.h"
#include "dsp/Mask.h"

#include "Error.h"
#include "stringtok.h"
#include "pad.h"

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>


using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::SingleThread::SingleThread ()
  : cerr( std::cerr.rdbuf() ), error (InvalidState, "")
{
  scratch = new Scratch;
  gpu_stream = undefined_stream;
  gpu_device = 0;
}

dsp::SingleThread::~SingleThread ()
{
}

void dsp::SingleThread::set_configuration (Config* configuration) try
{
  if (config == configuration)
    return;

  config = configuration;

  if (configuration && initialize_during_configuration)
    initialize ();
}
catch (Error& error)
{
  throw error += "SingleThread::set_configuration";
}

void dsp::SingleThread::take_ostream (std::ostream* newlog)
{
  if (newlog)
    this->cerr.rdbuf( newlog->rdbuf() );

  if (log)
    delete log;

  log = newlog;
}

//! Return true if this thread will run on the GPU
bool dsp::SingleThread::run_on_gpu() const
{
#if HAVE_CUDA
  return thread_id < config->get_cuda_ndevice();
#else
  return false;
#endif
}

dsp::Memory* dsp::SingleThread::get_memory()
{
  if (!device_memory)
    device_memory = new MemoryHost;
  return device_memory;
}

void* dsp::SingleThread::get_gpu_stream()
{
  return gpu_stream;
}

bool dsp::SingleThread::report_vitals () const
{
  return thread_id == 0 && config && config->report_vitals;
}

dsp::SingleThread::Config* dsp::SingleThread::get_configuration()
{
  return config;
}

void dsp::SingleThread::append(dsp::Operation* op)
{
  operations.push_back(op);
}

void dsp::SingleThread::initialize () try
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::initialize config=" << config.ptr() << endl;

#if HAVE_CUDA
  cudaStream_t stream = 0;
  if (run_on_gpu())
  {
    gpu_device = config->cuda_device[thread_id];

    // first construct message as a string to avoid confusing multi-threaded interleaved output at << breaks
    string message = app() + ": thread " + tostring(thread_id) + " using CUDA device " + tostring(gpu_device);
    cerr << message << endl;

    int ndevice = 0;
    cudaError err = cudaGetDeviceCount(&ndevice);

    if (err != cudaSuccess || gpu_device >= ndevice)
      throw Error (InvalidParam, "dsp::SingleThread::initialize",
                   "device=%d >= ndevice=%d cudaError=%s", gpu_device, ndevice, cudaGetErrorString(err));

    err = cudaSetDevice (gpu_device);
    if (err != cudaSuccess)
      throw Error (InvalidState, "dsp::SingleThread::initialize",
                   "cudaMalloc failed: %s", cudaGetErrorString(err));

    // always create a stream, even for 1 thread
    cudaStreamCreate( &stream );

    if (Operation::verbose)
      cerr << "dsp::SingleThread::initialize thread " << thread_id << " on stream " << stream << endl;

    gpu_stream = stream;
  }
#endif
}
catch (Error& error)
{
  throw error += "SingleThread::initialize";
}

//! Set the source from which data will be read
void dsp::SingleThread::set_source (Source* _source)
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_source source=" << (void*)_source << endl;

  source = _source;
}

dsp::Source* dsp::SingleThread::get_source ()
{
  return source;
}

void dsp::SingleThread::set_affinity (int core)
{
#if HAVE_SCHED_SETAFFINITY
  cpu_set_t set;
  CPU_ZERO (&set);
  CPU_SET (core, &set);

  pid_t tpid = syscall (SYS_gettid);

  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_affinity thread=" << thread_id << " tpid=" << tpid << " core=" << core << endl;

  if (sched_setaffinity(tpid, sizeof(cpu_set_t), &set) < 0)
    throw Error (FailedSys, "dsp::SingleThread::set_affinity", "sched_setaffinity (%d)", core);
#endif
}

//! Share any necessary resources with the specified thread
void dsp::SingleThread::share (SingleThread* other)
{
  colleague = other;

  typedef Transformation<TimeSeries,TimeSeries> Xform;

  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    Xform* trans0 = dynamic_kast<Xform>( other->operations[iop] );

    if (!trans0)
      continue;

    if (!trans0->has_buffering_policy())
      continue;

    InputBuffering::Share* ibuf0;
    ibuf0 = dynamic_cast<InputBuffering::Share*>( trans0->get_buffering_policy() );

    if (!ibuf0)
      continue;

    Xform* trans = dynamic_kast<Xform>( operations[iop] );

    if (!trans)
      throw Error (InvalidState, "dsp::SingleThread::share", "mismatched operation type");

    if (!trans->has_buffering_policy())
      throw Error (InvalidState, "dsp::SingleThread::share", "mismatched buffering policy");

    if (Operation::verbose)
      cerr << "dsp::SingleThread::share sharing buffering policy of " << trans->get_name() << endl;

    trans->set_buffering_policy( ibuf0->clone(trans) );
  }
}

dsp::TimeSeries* dsp::SingleThread::new_time_series ()
{
  config->buffers ++;

  if (config->weighted_time_series)
  {
    if (Operation::verbose)
      cerr << "Creating WeightedTimeSeries instance" << endl;
    return new WeightedTimeSeries;
  }
  else
  {
    if (Operation::verbose)
      cerr << "Creating TimeSeries instance" << endl;
    return new TimeSeries;
  }
}

void dsp::SingleThread::construct () try
{
  TimeSeries::auto_delete = false;

  operations.resize (0);

  // each timeseries created will be counted in new_time_series
  config->buffers = 0;

  if (thread_id < config->affinity.size())
    set_affinity (config->affinity[thread_id]);

  if (thread_id == 0)
    prepare (source);

  if (!source->has_output())
  {
    if (Operation::verbose)
      cerr << "dsp::SingleThread::construct set Source output to new TimeSeries" << endl;
    source->set_output (new_time_series());
  }
  else if (Operation::verbose)
  {
    cerr << "dsp::SingleThread::construct Source has output TimeSeries" << endl;
  }

  source_output = source->get_output();

  operations.push_back (source.get());

#if HAVE_CUDA

  if (run_on_gpu())
  {
    cudaStream_t stream = (cudaStream_t) gpu_stream;
    device_memory = new CUDA::DeviceMemory (stream, gpu_device);

    if (thread_id == 0)
    {
      if (config->input_buffering)
        cerr << app() << ": input_buffering enabled" << endl;
      else
        cerr << app() << ": input_buffering disabled" << endl;
    }

    if (source->get_device_supported( device_memory ))
    {
      if (thread_id == 0)
      {
        cerr << app() << ": manager supports device memory" << endl;

        if (!config->input_buffering)
        {
          // overlap memory on stream/device of thread_id 0
          configure_overlap (source, device_memory);
        }
      }
      source->set_device( device_memory );
      source_output->set_memory( device_memory );
      source_output->set_engine (new CUDA::TimeSeriesEngine (device_memory));
    }
    else
    {
      if (Operation::verbose && thread_id == 0)
        cerr << "SingleThread: unpack on CPU" << endl;

      TransferCUDA* transfer = new TransferCUDA (stream);
      transfer->set_kind( cudaMemcpyHostToDevice );
      transfer->set_input( source_output );

      source_output = new_time_series ();
      source_output->set_memory (device_memory);
      transfer->set_output( source_output );
      operations.push_back (transfer);
    }
    auto source_output_wts = dynamic_cast<dsp::WeightedTimeSeries *>(source_output.get());
    if (source_output_wts)
    {
      source_output_wts->set_weights_memory(device_memory);
    }
  }
  else
  {
    source->set_device( Memory::get_manager () );
  }

#else  // not HAVE_CUFFT

  source->set_device( Memory::get_manager () );

#endif // HAVE_CUFFT

#if HAVE_CFITSIO && HAVE_fits

  if (config->apply_FITS_scale_and_offset &&
      source->get_info()->get_machine() == "FITS")
  {
    auto manager = dynamic_cast<IOManager*>(source.get());

    if (!manager)
      throw Error (InvalidState, "dsp::SingleThread::construct",
        "Source is not of type IOManager - cannot apply_FITS_scale_and_offset");

    FITSUnpacker* fun = dynamic_cast<FITSUnpacker*> (manager->get_unpacker());
    fun->apply_scale_and_offset (true);
  }

#endif

}
catch (Error& error)
{
  throw error += "dsp::SingleThread::construct";
}

void dsp::SingleThread::prepare ()
{
  for (unsigned idump=0; idump < config->dump_before.size(); idump++)
    insert_dump_point (config->dump_before[idump]);

  // so that repeated calls to prepare do not insert more dump points
  config->dump_before.clear();

  for (unsigned imask=0; imask < config->mask_before.size(); imask++)
    insert_mask (config->mask_before[imask]);

  // so that repeated calls to prepare do not insert more mask points
  config->mask_before.clear();

  if (log)
    scratch->set_cerr (*log);

  // prepare all operations
  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (Operation::verbose)
      cerr << "dsp::SingleThread::prepare " << operations[iop]->get_name() << endl;

    if (log)
      operations[iop] -> set_cerr (*log);

    operations[iop]->prepare ();
  }

  prepare_block_size();

  if (thread_id == 0)
  {
    if (config->report_vitals)
    {
      double megabyte = 1024*1024;
      cerr << app() << ": blocksize=" << block_size
           << " samples or " << block_size_bytes/megabyte << " MB" << endl;
    }

    // seek_epoch after Operations are prepared and have computed delay times
    if (config->seek_epoch != 0.0)
    {
      seek_epoch (config->seek_epoch);
    }
  }
}

void dsp::SingleThread::prepare_block_size()
{
  if (!block_size)
  {
    if (Operation::verbose)
      cerr << "dsp::SingleThread::prepare_block_size block_size not set; calling set_block_size" << endl;
    set_block_size();
  }
  else
  {
    if (Operation::verbose)
      cerr << "dsp::SingleThread::prepare_block_size block_size already set - returning" << endl;
  }
}

void dsp::SingleThread::seek_epoch(const MJD& epoch)
{
  double total_delay = 0;

  for (auto& op: operations)
  {
    double delay = op->get_delay_time();
    if (Operation::verbose)
      cerr << "dsp::SingleThread::seek_epoch " << op->get_name() << " induces " << delay*1e3 << " ms of delay" << endl;
    total_delay += delay;
  }

  Observation *info = source->get_info();
  double seek_seconds = (epoch - info->get_start_time()).in_seconds();
  seek_seconds -= total_delay;

  if (Operation::verbose)
    std::cerr << "dsp::SingleThread::seek_epoch seek=" << seek_seconds*1e3 << " ms" << endl;

  source->seek_time (seek_seconds);
}

void dsp::SingleThread::set_block_size (uint64_t minimum_samples, uint64_t input_overlap)
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_block_size minimum_samples=" << minimum_samples << " input_overlap=" << input_overlap << endl;

  const float megabyte = 1024 * 1024;

  if (input_overlap > minimum_samples)
    throw Error (InvalidParam, "dsp::SingleThread::set_block_size",
                 "input_overlap=%u is greater than minimum_samples=%u", input_overlap, minimum_samples);

  unsigned nblocks = config->get_times_minimum_ndat();
  minimum_samples = (minimum_samples - input_overlap) * nblocks + input_overlap;

  uint64_t total_storage = 0;
  uint64_t max_scratch = 0;

  for (auto& op: operations)
  {
    uint64_t storage = op->bytes_storage();
    if (Operation::verbose)
      cerr << "dsp::SingleThread::set_block_size " << op->get_name() << " requires " << storage << " bytes of storage" << endl;
    total_storage += storage;

    uint64_t scratch = op->bytes_scratch();
    if (Operation::verbose)
      cerr << "dsp::SingleThread::set_block_size " << op->get_name() << " requires " << scratch << " bytes of scratch" << endl;
    max_scratch = std::max(max_scratch,scratch);
  }

  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_block_size " << operations.size() <<
    " operations require " << total_storage/megabyte << " MB total storage"
    " and " << max_scratch/megabyte << " MB maximum scratch" << endl;

  uint64_t additional_bytes = total_storage + max_scratch;

  unsigned nbit  = source->get_info()->get_nbit();
  unsigned ndim  = source->get_info()->get_ndim();
  unsigned npol  = source->get_info()->get_npol();
  unsigned nchan = source->get_info()->get_nchan();
  unsigned copies = config->get_nbuffers();

  // each nbit number will be unpacked into a float
  double nbyte = double(nbit)/8 + copies * sizeof(float);

  double nbyte_dat = nbyte * ndim * npol * nchan;

  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_block_size copies=" << copies
         << " nbit=" << nbit << " nbyte=" << nbyte << " total=" << nbyte_dat << endl;

  if (config->get_minimum_RAM() > additional_bytes)
  {
    block_size = (config->get_minimum_RAM() - additional_bytes) / nbyte_dat;
    if (Operation::verbose)
      cerr << "dsp::SingleThread::set_block_size minimum block_size=" << block_size << endl;
  }

  if (config->get_maximum_RAM() > additional_bytes)
  {
    block_size = (config->get_maximum_RAM() - additional_bytes) / nbyte_dat;
    if (Operation::verbose)
      cerr << "dsp::SingleThread::set_block_size maximum block_size=" << block_size << endl;
  }

  if (block_size == 0 || block_size < minimum_samples)
  {
    float min_ram = ceil (minimum_samples * nbyte_dat + additional_bytes);

    /* Printing out the exact minimum RAM requirements requires too many digits of precision.
      Therefore, round up to the nearest integer MB. */
    unsigned min_ram_MB = ceil (min_ram/megabyte);
    if (Operation::verbose)
      cerr << "dsp::SingleThread::set_block_size insufficient RAM" << endl;

    throw Error (InvalidState, "dsp::SingleThread::set_block_size",
                 "insufficient RAM: limit=%g MB -> block=" UI64 " samples\n\t"
                 "require=" UI64 " samples -> a minimum of \"-U %u\" on command line",
                 float(config->get_maximum_RAM())/megabyte, block_size,
                 minimum_samples, min_ram_MB);
  }

  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_block_size calling Source::set_block_size " << block_size << endl;

  source->set_overlap(input_overlap);
  source->set_block_size(block_size);

  uint64_t actual_block_size = source->get_block_size();
  if (Operation::verbose)
    cerr << "dsp::SingleThread::set_block_size actual_block_size " << actual_block_size << endl;

  block_size_bytes = actual_block_size * nbyte_dat + additional_bytes;
}

void dsp::SingleThread::prepare (Source* source)
{
  config->prepare(source);
}

void dsp::SingleThread::configure_overlap (Source* source, Memory* device_memory)
{
  auto input_source = dynamic_cast<dsp::InputSource<Input>*> (source);

  if (!input_source)
    throw Error (InvalidState, "dsp::SingleThread::configure_overlap", "Source does not have an Input");

  dsp::Seekable * seekable = dynamic_cast<dsp::Seekable*>( input_source->get_input() );
  if (!seekable)
    throw Error (InvalidState, "dsp::SingleThread::configure_overlap", "Input is not Seekable");

  cerr << app() << ": disabling input buffering, using overlap memory instead" << endl;
  seekable->set_overlap_buffer_memory (device_memory);
}

void dsp::SingleThread::insert_dump_point (const std::string& transform_name)
{
  typedef HasInput<TimeSeries> Xform;

  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (operations[iop]->get_name() == transform_name)
    {
      Xform* xform = dynamic_cast<Xform*>( operations[iop].get() );
      if (!xform)
        throw Error (InvalidParam, "dsp::SingleThread::insert_dump_point",
                     transform_name + " does not have TimeSeries input");

      string filename = "pre_" + transform_name;

      if (config->get_total_nthread() > 1)
        filename += "." + tostring (thread_id);

      filename += ".dump";

      cerr << app() << ": dump output in " << filename << endl;

      Dump* dump = new Dump;
      dump->set_output( fopen(filename.c_str(), "w") );
      dump->set_input( xform->get_input() ) ;
      dump->set_output_binary (true);

      operations.insert (operations.begin()+iop, dump);
      iop++;
    }
  }
}

void dsp::SingleThread::insert_mask (const std::string& description)
{
  typedef HasInput<TimeSeries> Xform;

  string mask_description = description;
  string transform_name = stringtok(mask_description, ":");

  if (Operation::verbose)
    cerr << "dsp::SingleThread::insert_mask mask_description=" << mask_description
         << " before xform=" << transform_name << endl;

  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (operations[iop]->get_name() == transform_name)
    {
      Xform* xform = dynamic_cast<Xform*>( operations[iop].get() );
      if (!xform)
        throw Error (InvalidParam, "dsp::SingleThread::insert_mask",
                     transform_name + " does not have TimeSeries input");

      Reference::To<Mask> mask = Mask::factory(mask_description);

      cerr << app() << ": inserting " << mask->get_name() << " before " << transform_name << endl;

      mask->set_input( xform->get_input() ) ;
      operations.insert (operations.begin()+iop, mask.get());
      iop++;

      return;
    }
  }

  // named operation not found
  throw Error (InvalidParam, "dsp::SingleThread::insert_mask", "no operation named '" + transform_name + "' in pipeline");
}

const std::string& dsp::SingleThread::app()
{
  if (!config)
    throw Error (InvalidState, "dsp::SingleThread::app", "no configuration");
  return config->application_name;
}

uint64_t dsp::SingleThread::get_minimum_samples () const
{
  return minimum_samples;
}

//! Run through the data
void dsp::SingleThread::run () try
{
  if (Operation::verbose)
  {
    cerr << "dsp::SingleThread::run this=" << this
         << " nops=" << operations.size() << endl;

    for (unsigned iop=0; iop < operations.size(); iop++)
    {
      cerr << "dsp::SingleThread::run operation (" << iop << "): " << operations[iop]->get_name() << endl;
    }
  }

  // ensure that all operations are using the local log and scratch space
  for (unsigned iop=0; iop < operations.size(); iop++)
  {
    if (!operations[iop] -> scratch_was_set ())
      operations[iop] -> set_scratch (scratch);

    operations[iop] -> reserve ();
  }

  int64_t last_decisecond = -1;

  bool finished = false;
  while (!finished)
  {
    while (!finished && !source->end_of_data())
    {
      optime.start();
      for (unsigned iop=0; iop < operations.size(); iop++) try
      {
        if (Operation::verbose)
          cerr << "dsp::SingleThread::run calling " << operations[iop]->get_name() << endl;

        operations[iop]->operate ();

        if (Operation::verbose)
          cerr << "dsp::SingleThread::run "
              << operations[iop]->get_name() << " done" << endl;
      }
      catch (Error& error)
      {
        if (error.get_code() == EndOfFile)
        {
          if (Operation::verbose)
            cerr << "dsp::SingleThread::run EndOfFile exception raised" << endl;
          finished = true;
          break;
        }
        end_of_data ();

        throw error += "dsp::SingleThread::run";
      }
      optime.stop();

      if (thread_id==0 && config->report_done)
      {
        double seconds = source->get_current_time();
        int64_t decisecond = int64_t( seconds * 10 );

        if (decisecond > last_decisecond)
        {
          last_decisecond = decisecond;
          cerr << "Finished " << decisecond/10.0 << " s";

          if (source->get_total_samples())
          {
            float frac_done = float(source->get_current_sample()) / source->get_total_samples();
            cerr << " (" << int (100.0*frac_done) << "%)";
            cerr << "   \r";
          }
        }
      }
    }
    finished = true;

    if (config->run_repeatedly)
    {
      ThreadContext::Lock context (source_context);

      if (config->repeated == 0 && source->get_current_sample() != 0)
      {
        finished = false;
        source->restart();
        config->repeated = 1;

        prepare(source);
      }
      else if (config->repeated)
      {
        config->repeated ++;
        finished = false;

        if (config->repeated == config->get_total_nthread())
          config->repeated = 0;
      }
    }
  }

  if (Operation::verbose)
    cerr << "dsp::SingleThread::run end of data id=" << thread_id << endl;

  end_of_data ();

  if (Operation::verbose)
    cerr << "dsp::SingleThread::run exit" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SingleThread::run";
}

bool same_name (const dsp::Operation* A, const dsp::Operation* B)
{
  return A->get_name() == B->get_name();
}

template<typename C>
unsigned find_name (const C& container, unsigned i, const dsp::Operation* B)
{
  while (i < container.size() && ! same_name(container[i], B))
    i++;
  return i;
}

void dsp::SingleThread::combine (const SingleThread* that)
{
  if (Operation::verbose)
    cerr << "dsp::SingleThread::combine"
         << " this size=" << operations.size()
         << " ptr=" << &(this->operations)
         << " that size=" << that->operations.size()
         << " ptr=" << &(that->operations) << endl;

  unsigned ithis = 0;
  unsigned ithat = 0;

  while (ithis < operations.size() && ithat < that->operations.size())
  {
    if (! same_name(operations[ithis], that->operations[ithat]))
    {
      // search for that in this
      unsigned jthis = find_name (operations, ithis, that->operations[ithat]);
      if (jthis == operations.size())
      {
        if (Operation::verbose)
          cerr << "dsp::SingleThread::combine insert "
               << that->operations[ithat]->get_name() << endl;

        // that was not found in this ... insert it and skip it
        operations.insert( operations.begin()+ithis, that->operations[ithat] );
        ithis ++;
        ithat ++;
      }
      else
      {
        // that was found later in this ... skip to it
        ithis = jthis;
      }

      continue;

#if 0
      if (operations[ithis]->get_function() != Operation::Procedural)
      {
        ithis ++;
        continue;
      }

      if (that->operations[ithat]->get_function() != Operation::Procedural)
      {
        ithat ++;
        continue;
      }

      throw Error (InvalidState, "dsp::SingleThread::combine",
                   "operation names do not match "
                   "'"+ operations[ithis]->get_name()+"'"
                   " != '"+that->operations[ithat]->get_name()+"'");
#endif
    }

    if (Operation::verbose)
      cerr << "dsp::SingleThread::combine "
           << operations[ithis]->get_name() << endl;

    operations[ithis]->combine( that->operations[ithat] );

    ithis ++;
    ithat ++;
  }

  if (ithis != operations.size() || ithat != that->operations.size())
    throw Error (InvalidState, "dsp::SingleThread::combine",
                 "processes have different numbers of operations");
}

//! Optionally print a wall time report
void dsp::SingleThread::finish () try
{
  source->close();

  if (Operation::record_time)
    for (unsigned iop=0; iop < operations.size(); iop++)
      operations[iop]->report();

  // ensure that the source is deleted
  // this kludge works around any potential circular reference that might
  // incorrectly stop the destructor from being called
  if (source->__is_on_heap())
    delete source;
}
catch (Error& error)
{
  throw error += "dsp::SingleThread::finish";
}

void dsp::SingleThread::end_of_data ()
{
  // do nothing
}

//! Create new Source based on command line options
dsp::Source* dsp::SingleThread::Config::open (int argc, char** argv)
{
  SourceOpen open;
  open.command_line_header = command_line_header;
  open.force_contiguity = force_contiguity;

  return open.open(argc,argv);
}

void dsp::SingleThread::Config::prepare (Source* source)
{
  if (list_attributes || editor.will_modify())
    cout << editor.process (source->get_info()) << endl;

  if (seek_seconds)
  {
    if (Operation::verbose)
      std::cerr << "dsp::SingleThread::Config::prepare seek_seconds=" << seek_seconds << endl;

    source->seek_time (seek_seconds);
  }

  if (total_seconds)
  {
    if (Operation::verbose)
      std::cerr << "dsp::SingleThread::Config::prepare total_seconds=" << total_seconds << endl;

    source->set_total_time (seek_seconds + total_seconds);
  }

  auto manager = dynamic_cast<IOManager*>(source);
  if (manager)
    twobit_config.configure( manager->get_unpacker() );

  if (source_prepare)
    source_prepare( source );
}

//! set the number of CPU threads to be used
void dsp::SingleThread::Config::set_nthread (unsigned cpu_nthread)
{
  nthread = cpu_nthread;
}

//! get the total number of threads
unsigned dsp::SingleThread::Config::get_total_nthread () const
{
  unsigned total_nthread = nthread + get_cuda_ndevice();

  if (total_nthread)
    return total_nthread;

  return 1;
}

// set the cuda devices to be used
void dsp::SingleThread::Config::set_cuda_device (string txt)
{
  while (txt != "")
  {
    string dev = stringtok (txt, ",");
    cuda_device.push_back( fromstring<unsigned>(dev) );
  }
}

// set the cpu on which threads will run
void dsp::SingleThread::Config::set_affinity (string txt)
{
  while (txt != "")
  {
    string cpu = stringtok (txt, ",");
    affinity.push_back( fromstring<unsigned>(cpu) );
  }
}

// set block size to this factor times the minimum possible
void dsp::SingleThread::Config::set_times_minimum_ndat (unsigned ndat)
{
  times_minimum_ndat = ndat;
  maximum_RAM = 0;
}

// set block_size to result in at least this much RAM usage
void dsp::SingleThread::Config::set_minimum_RAM (uint64_t ram)
{
  minimum_RAM = ram;
  maximum_RAM = 0;
  times_minimum_ndat = 1;
}

// set block_size to result in approximately this much RAM usage
void dsp::SingleThread::Config::set_maximum_RAM (uint64_t ram)
{
  maximum_RAM = ram;
  minimum_RAM = 0;
  times_minimum_ndat = 1;
}

void dsp::SingleThread::Config::set_minimum_RAM_MB(const std::string& ram_min)
{
  if (ram_min.empty())
    return;

  double MB = fromstring<double> (ram_min);
  set_minimum_RAM(uint64_t( MB * 1024.0 * 1024.0 ));
}

void dsp::SingleThread::Config::set_maximum_RAM_MB(const std::string& ram_limit)
{
  if (ram_limit.empty())
    return;

  if (ram_limit == "min")
  {
    set_times_minimum_ndat( 1 );
  }
  else
  {
    unsigned times = 0;
    if ( sscanf(ram_limit.c_str(), "minX%u", &times) == 1 )
    {
      set_times_minimum_ndat( times );
    }
    else
    {
      double MB = fromstring<double> (ram_limit);
      set_maximum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
    }
  }
}

void dsp::SingleThread::Config::twobit_parse (const std::string& text)
{
  std::cerr << application_name << ": " << twobit_config.parse (text) << endl;
}

void dsp::SingleThread::Config::list_backends()
{
  cout << application_name << endl;
  SourceOpen::list_backends();
}

//! Add command line options
void dsp::SingleThread::Config::add_options (CommandLine::Menu& menu)
{
  CommandLine::Argument* arg;

  arg = menu.add (this, &Config::set_quiet, 'q');
  arg->set_long_name ("Q");
  arg->set_help ("quiet mode");

  arg = menu.add (this, &Config::set_verbose, 'v');
  arg->set_help ("verbose mode");

  arg = menu.add (this, &Config::set_very_verbose, 'V');
  arg->set_help ("very verbose mode");

  menu.add ("\n" "Input handling options:");

  arg = menu.add (apply_FITS_scale_and_offset, "scloffs");
  arg->set_help ("denormalize PSRFITS using DAT_SCL and DAT_OFFS");

  arg = menu.add (this, &Config::twobit_parse, '2', "code");
  arg->set_help ("unpacker options (\"2-bit\" excision)");
  arg->set_long_help (twobit_config.help ("2"));

  arg = menu.add (input_buffering, "overlap");
  arg->set_help ("disable input buffering");

  arg = menu.add (force_contiguity, "cont");
  arg->set_help ("assume that input files are contiguous");

  arg = menu.add (run_repeatedly, "repeat");
  arg->set_help ("repeatedly read from input until an empty is encountered");

  arg = menu.add (optimal_order, "order");
  arg->set_help ("order data optimally when possible [default:true]");

  arg = menu.add (seek_epoch, "seek", "MJD");
  arg->set_help ("seek such that the first output time sample occurs at MJD");
  arg->set_long_help
    (" Any time delays introduced by operations are taken into account \n");

  arg = menu.add (seek_seconds, 'S', "seek");
  arg->set_help ("start processing at t=seek seconds");

  arg = menu.add (total_seconds, 'T', "total");
  arg->set_help ("process only t=total seconds");

  arg = menu.add (&editor, &TextEditor<Observation>::add_commands, "set", "key=value");
  arg->set_help ("set observation attributes");

  arg = menu.add (list_attributes, "list");
  arg->set_help ("list observation attributes");

  arg = menu.add (command_line_header, "header");
  arg->set_help ("command line arguments are header values (not filenames)");

  if (weighted_time_series)
  {
    arg = menu.add (weighted_time_series, 'W');
    arg->set_help ("disable weights (allow bad data)");
  }

  arg = menu.add (this, &SingleThread::Config::list_backends, "backends");
  arg->set_help ("list backend handlers");

  menu.add ("\n" "Memory and processing options:");

  arg = menu.add (this, &Config::set_minimum_RAM_MB, "minram", "MB");
  arg->set_help ("minimum RAM usage in MB");

  string ram_limit;
  arg = menu.add (this, &Config::set_maximum_RAM_MB, 'U', "MB|minX");
  arg->set_help ("upper limit on RAM usage");
  arg->set_long_help
    ("specify either the floating point number of megabytes; e.g. -U 256 \n"
     "or a multiple of the minimum possible block size; e.g. -U minX2 \n");

  if (can_thread)
  {
    arg = menu.add (this, &Config::set_nthread, "threads", "nthread");
    arg->set_help ("number of CPU processor threads");
  }

#if HAVE_SCHED_SETAFFINITY
  arg = menu.add (this, &Config::set_affinity, "cpu", "cores");
  arg->set_help ("comma-separated list of CPU cores");
#endif

#if HAVE_CUFFT
  if (can_cuda)
  {
    arg = menu.add (this, &Config::set_cuda_device, "cuda", "devices");
    arg->set_help ("comma-separated list of CUDA devices");
  }
#endif

  arg = menu.add (this, &Config::set_fft_library, 'Z', "lib");
  arg->set_help ("choose the FFT library ('-Z help' for availability)");

  dsp::Operation::report_time = false;

  arg = menu.add (dsp::Operation::record_time, 'r');
  arg->set_help ("report time spent performing each operation");

  arg = menu.add (dump_before, "dump", "op");
  arg->set_help ("dump time series before performing operation");

  arg = menu.add (mask_before, "mask", "op");
  arg->set_help ("mask time series before performing operation");
}

void dsp::SingleThread::Config::set_quiet ()
{
  dsp::set_verbosity (0);
  report_vitals = false;
  report_done = false;
}

void dsp::SingleThread::Config::set_verbose ()
{
  dsp::set_verbosity (2);
}

void dsp::SingleThread::Config::set_very_verbose ()
{
  dsp::set_verbosity (3);
}

#include "FTransform.h"
#include <stdlib.h>

void dsp::SingleThread::Config::set_fft_library (string fft_lib)
{
  if (fft_lib == "help")
  {
    unsigned nlib = FTransform::get_num_libraries ();

    if (nlib == 1)
      std::cerr << "There is 1 available FFT library: " << FTransform::get_library_name (0) << endl;
    else
    {
      std::cerr << "There are " << nlib << " available FFT libraries:";
      for (unsigned ilib=0; ilib < nlib; ilib++)
        std::cerr << " " << FTransform::get_library_name (ilib);

      std::cerr << "\nThe default FFT library is " << FTransform::get_library() << endl;
    }
    exit (0);
  }
  else if (fft_lib == "simd")
    FTransform::simd = true;
  else
  {
    FTransform::set_library (fft_lib);
    std::cerr << "FFT library set to " << fft_lib << endl;
  }
}

const dsp::Pipeline::PerformanceMetrics* dsp::SingleThread::get_performance_metrics()
{
  // determine the total data time and bytes processed from the source only
  performance_metrics.total_data_time = source->get_performance_metrics()->total_data_time;
  performance_metrics.total_bytes_processed = source->get_performance_metrics()->total_bytes_processed;

  // the total pipeline processing time will be the sum of the operation's processing times
  performance_metrics.total_processing_time = 0;

  std::vector<std::pair<std::string, dsp::Operation::PerformanceMetrics>> operation_metrics = {};
  for (auto operation : get_operations())
  {
    auto *op_metrics = operation->get_performance_metrics();
    if (op_metrics != nullptr)
    {
      operation_metrics.push_back({operation->get_name(), *op_metrics});
      performance_metrics.total_processing_time += op_metrics->total_processing_time;
    }
  }
  performance_metrics.operation_metrics = operation_metrics;

  return &performance_metrics;
}
