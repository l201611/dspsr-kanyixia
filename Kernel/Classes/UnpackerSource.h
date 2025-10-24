//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/UnpackerSource.h

#ifndef __dsp_Kernel_Classes_UnpackerSource_h
#define __dsp_Kernel_Classes_UnpackerSource_h

#include "dsp/OperationPerformanceMetrics.h"
#include "dsp/InputSource.h"

namespace dsp {

  //! Produces TimeSeries data by integrating an InputType with an UnpackerType
  template<class UnpackerType,class InputType>
  class UnpackerSource : public InputSource<InputType>
  {

  public:

    typedef typename InputType::OutputType BitContainerType;

    //! Constructor
    UnpackerSource (const char* name) : InputSource<InputType>(name) {}

    //! Return true if the source supports the specified output order
    bool get_order_supported (TimeSeries::Order) const override;

    //! Set the order of the dimensions in the output TimeSeries
    void set_output_order (TimeSeries::Order) override;

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const override;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*) override;

    //! Prepare the input and unpacker
    void prepare () override;

    //! Reserve the maximum amount of output space required
    void reserve () override;

    //! Add any input and unpacker extensions
    void add_extensions (Extensions*) override;

    //! Combine the input and unpacker
    void combine (const Operation*) override;

    //! Reset the input and unpacker
    void reset () override;

    //! Reset the accumulated wall time of the unpacker
    /*! Accumulated wall time of input is not reset because it is typically shared between threads,
        and this method was introduced only to eliminate incorrect multiple counting of time. */
    void reset_time () const override;

    //! Report operation statistics
    void report () const override;

    /**
     * @brief Get the performance metrics for the UnpackerSource.
     *
     * @note The processing time metric only includes the contribution from the
     * Unpacker, excluding the time spent in the Input. This means that any contribution
     * from I/O of the Input or memory copying to the BitSeries are not included.
     *
     * @return const Operation::PerformanceMetrics* performance metrics of the Unpacker
     */
    const Operation::PerformanceMetrics* get_performance_metrics () override;

    //! The number of bytes of additional storage used by the operation
    uint64_t bytes_storage() const override;

    //! The number of bytes of scratch space used by the operation
    uint64_t bytes_scratch () const override;

    //! Return pointer to the appropriate InputType
    const InputType* get_input () const override { return input; }
    InputType* get_input () override { return input; }

    //! Set the InputType operator (should not normally need to be used)
    void set_input (InputType* input) override;

    //! Return pointer to the appropriate UnpackerType
    const UnpackerType* get_unpacker () const;
    UnpackerType* get_unpacker ();

    //! Set the UnpackerType (should not normally need to be used)
    void set_unpacker (UnpackerType* unpacker);

    //! Set the TimeSeries object used to store output data
    void set_output (TimeSeries* output) override;

    //! Get the TimeSeries object used to store output data
    TimeSeries* get_output () override;

    //! Return true if this object has a TimeSeries object to store output data
    bool has_output () const override;

    //! Set the container into which intermediate raw data will be loaded
    // (should not normally need to be used)
    virtual void set_bit_container (BitContainerType*);

    //! Set custom post load operation
    void set_post_load_operation (Operation * op);

    //! Get the minimum number of time samples that can be output by the source
    uint64_t get_resolution () const override;

    //! Load and convert the next block of data
    virtual void load (TimeSeries* data);

    //! Set the scratch space
    void set_scratch (Scratch* scratch) override;

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const override;

  protected:

    //! Load the TimeSeries specified with set_output
    void operation () override;

    //! Appropriate InputType subclass
    Reference::To<InputType> input;

    //! Appropriate UnpackerType subclass
    Reference::To<UnpackerType> unpacker;

    //! The container in which the intermediate raw data is loaded
    Reference::To<BitContainerType> bit_container;

    //! The container in which the TimeSeries data is unpacked
    Reference::To<TimeSeries> output;

    //! Optional operation performed between load and unpack
    Reference::To<Operation> post_load_operation;
  };
}

template<class UnT,class InT>
bool dsp::UnpackerSource<UnT,InT>::get_device_supported (Memory* device_memory) const
{
  return unpacker->get_device_supported(device_memory);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_device (Memory* device_memory)
{
  if (!unpacker)
    throw Error (InvalidState, "dsp::UnpackerSource<UnT,InT>::set_device", "Unpacker not set");

  if (!unpacker->get_device_supported(device_memory))
    throw Error (InvalidParam, "dsp::UnpackerSource<UnT,InT>::set_device", "Memory not supported by Unpacker");

  unpacker->set_device( device_memory );

  if (!bit_container)
    set_bit_container (new BitContainerType);

  bit_container->set_memory( device_memory );
}

template<class UnT,class InT>
bool dsp::UnpackerSource<UnT,InT>::get_order_supported (TimeSeries::Order order) const
{
  if (!unpacker)
    throw Error (InvalidState, "dsp::UnpackerSource<UnT,InT>::get_order_supported", "Unpacker not set");

  return unpacker->get_order_supported(order);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_output_order (TimeSeries::Order order)
{
  if (!unpacker)
    throw Error (InvalidState, "dsp::UnpackerSource<UnT,InT>::set_output_order", "Unpacker not set");

  unpacker->set_output_order(order);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_bit_container (BitContainerType* raw)
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::set_bit_container ptr=" << raw << std::endl;

  bit_container = raw;

  if (unpacker)
  {
    if (Operation::verbose)
      this->cerr << "dsp::UnpackerSource<UnT,InT>::set_bit_container call Unpacker::set_input" << std::endl;
    unpacker -> set_input (raw);
  }
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_output (TimeSeries* _output)
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::set_output (TimeSeries*) " << _output << std::endl;

  output = _output;

  if (unpacker)
  {
    if (Operation::verbose)
      this->cerr << "dsp::UnpackerSource<UnT,InT>::set_output call Unpacker::set_output" << std::endl;
    unpacker -> set_output (_output);
  }
}

template<class UnT,class InT>
dsp::TimeSeries* dsp::UnpackerSource<UnT,InT>::get_output ()
{
  return output;
}

template<class UnT,class InT>
bool dsp::UnpackerSource<UnT,InT>::has_output () const
{
  return output;
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_input (InT* _input)
{
  input = _input;

  if (!input)
    return;

  if (!unpacker || !unpacker->matches (input->get_info()))
    set_unpacker ( UnT::create( input->get_info() ) );
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_unpacker (UnT* _unpacker)
{
  unpacker = _unpacker;

  if (unpacker)
  {
    if (bit_container)
      unpacker -> set_input (bit_container);
    if (output)
      unpacker -> set_output (output);
  }
}

template<class UnT,class InT>
const UnT* dsp::UnpackerSource<UnT,InT>::get_unpacker () const
{
  return unpacker;
}

template<class UnT,class InT>
UnT* dsp::UnpackerSource<UnT,InT>::get_unpacker ()
{
  return unpacker;
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::load (TimeSeries* _output)
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::load (TimeSeries* = " << _output << ")" << std::endl;

  set_output (_output);

  operation ();
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::prepare ()
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::prepare" << std::endl;

  if (!bit_container)
    set_bit_container (new BitContainerType);

  input->set_output( bit_container );

  input->prepare();
  unpacker->prepare();

  unpacker->match_resolution (input);

  if (post_load_operation)
    post_load_operation->prepare ();

  this->prepared = true;
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::reserve ()
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::reserve" << std::endl;

  if (!bit_container)
    set_bit_container (new BitContainerType);

  input->reserve( bit_container );
  unpacker->reserve();
  if (post_load_operation)
    post_load_operation->reserve ();
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::add_extensions (Extensions* ext)
{
  if (input)
    input->add_extensions (ext);
  if (unpacker)
    unpacker->add_extensions (ext);
  if (post_load_operation)
    post_load_operation->add_extensions (ext);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::combine (const Operation* other)
{
  Operation::combine (other);

  const UnpackerSource<UnT,InT>* like = dynamic_cast<const UnpackerSource<UnT,InT>*>( other );
  if (!like)
    return;

  input->combine (like->input);
  unpacker->combine (like->unpacker);
  if (post_load_operation)
    post_load_operation->combine (like->post_load_operation);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::reset ()
{
  Operation::reset ();

  input->reset ();
  unpacker->reset ();
  if (post_load_operation)
    post_load_operation->reset ();
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::reset_time () const
{
  Operation::reset_time ();

  unpacker->reset_time ();
  if (post_load_operation)
    post_load_operation->reset_time ();
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::report () const
{
  if (input)
    input->report ();
  if (unpacker)
    unpacker->report ();
  if (post_load_operation)
    post_load_operation->report ();
}

template<class UnT,class InT>
const dsp::Operation::PerformanceMetrics* dsp::UnpackerSource<UnT,InT>::get_performance_metrics()
{
  return unpacker->get_performance_metrics();
}

template<class UnT,class InT>
uint64_t dsp::UnpackerSource<UnT,InT>::bytes_storage() const
{
  return input->bytes_storage() + unpacker->bytes_storage();
}

template<class UnT,class InT>
uint64_t dsp::UnpackerSource<UnT,InT>::bytes_scratch () const
{
  return std::max(input->bytes_scratch(), unpacker->bytes_scratch());
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::operation ()
{
  if (!bit_container)
    set_bit_container (new BitContainerType);

  input->load (bit_container);
  this->performance_metrics->update_metrics(bit_container);

  if (post_load_operation)
  {
    if (Operation::verbose)
      this->cerr << "dsp::UnpackerSource<UnT,InT>::operation post_load_operation->operate()" << std::endl;
    post_load_operation->operate ();
  }

  if (!output)
    return;

  unpacker->operate ();
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_post_load_operation (Operation * op)
{
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::set_post_load_operation(" << op << ")" << std::endl;
  post_load_operation = op;
}

template<class UnT,class InT>
uint64_t dsp::UnpackerSource<UnT,InT>::get_resolution () const
{
  unsigned resolution = input->get_resolution();
  if (Operation::verbose)
    this->cerr << "dsp::UnpackerSource<UnT,InT>::get_resolution input resolution=" << resolution << std::endl;

  if (unpacker->get_resolution())
  {
    resolution = unpacker->get_resolution();
    if (Operation::verbose)
      this->cerr << "dsp::UnpackerSource<UnT,InT>::get_resolution unpacker resolution=" << resolution << std::endl;
  }

  // ensure that the block size is a multiple of four
  if (resolution % 4)
  {
    if (resolution % 2 == 0)
      resolution *= 2;
    else
      resolution *= 4;
  }

  return resolution;
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_scratch (Scratch* scratch)
{
  if (Operation::verbose)
    this->cerr << "dsp::dsp::UnpackerSource<UnT,InT>::set_scratch" << std::endl;

  Operation::set_scratch(scratch);

  if (input && !input->has_context())
    input->set_scratch(scratch);

  if (unpacker)
    unpacker->set_scratch(scratch);
}

template<class UnT,class InT>
void dsp::UnpackerSource<UnT,InT>::set_cerr (std::ostream& os) const
{
  Operation::set_cerr( os );

  if (Operation::verbose)
    this->cerr << "dsp::dsp::UnpackerSource<UnT,InT>::set_cerr" << std::endl;

  if (input && !input->has_context())
    input->set_cerr( os );

  if (bit_container)
    bit_container->set_cerr( os );

  if (unpacker)
    unpacker->set_cerr( os );
}

#endif // !defined(__dsp_Kernel_Classes_UnpackerSource_h)
