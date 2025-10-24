//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Formats_ska1_SKAParallelUnpacker
#define __dsp_Kernel_Formats_ska1_SKAParallelUnpacker

#include "dsp/ParallelUnpacker.h"
#include <assert.h>

namespace dsp {

  /**
   * @brief Processes ParallelBitSeries with containing Data and Weights with a Single Unpacker
   *        This unpacker only supports FPT ordered timeseries output containers
   *
   */
  class SKAParallelUnpacker : public ParallelUnpacker
  {
  public:

    //! Constructor
    SKAParallelUnpacker ();

    //! Destructor
    ~SKAParallelUnpacker ();

    void configure (const Observation* observation);

    //! Return true if descriptor describes a file that can be opened
    bool matches (const Observation* observation) const override;

    //! Specialize the unpackers for the Observation
    void match (const Observation* observation) override;

    //! Reserve the maximum amount of space required in the output
    /*! If the output is a WeightedTimeSeries, set it up to store the weights */
    void reserve () override;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*) override;

    //! Engine used to perform unpacking on a device
    class Engine;
    void set_engine (Engine*);

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const override;

    //! The unpacking routine
    void unpack () override;

    //! Copy the input attributes to the output
    void prepare () override;

  private:

    //! Return the scale factor packed into the weights array, corresponding to the provided packet number
    float get_scale_factor(const unsigned char * weights, uint32_t packet_number);

    /**
     * @brief Templated method to unpack 8 or 16 bit integers from data and weights pointers FPT ordered time-series
     * The unpacked output TimeSeries must have been resized before caling this method.
     *
     * @tparam T type of input data to unpack [int8_t or int16_t]
     * @param in pointer to the input data array
     * @param weights pointer to the input weights array
     * @param nheaps number of packed heaps to unpack.
     */
    template <typename T>
    void unpack_samples(const T* in, const unsigned char * weights, uint32_t nheaps)
    {
      const uint32_t nval_per_packet = nsamp_per_packet * ndim;
      const uint32_t nsamp = nheaps * nsamp_per_packet;

      assert (output->get_ndat() == nsamp);
      assert (output->get_nchan() == nchan);
      assert (output->get_ndim() == ndim);
      assert (nchan == npackets_per_heap * nchan_per_packet);

      // Unpack quantised data store in heap, packet, pol, chan_block, samp_block ordering used in CBF/PSR formats
      uint32_t packet_number = 0;
      for (uint32_t iheap=0; iheap<nheaps; iheap++)
      {
        // number of values that are offset for the heap
        const uint32_t heap_val_offset = (iheap * nsamp_per_packet) * ndim;

        for (uint32_t ipacket=0; ipacket<npackets_per_heap; ipacket++)
        {
          const float scale_factor = get_scale_factor(weights, packet_number);
          const float multiplier = (std::isnan(scale_factor) || scale_factor<=0.0) ? 0.0 : 1.0 / scale_factor;

          if (std::isnan(scale_factor))
          {
            invalid_packets++;
          }

          for (uint32_t ichan=0; ichan<nchan_per_packet; ichan++)
          {
            const uint32_t ochan = (ipacket * nchan_per_packet) + ichan;
            for (uint32_t ipol=0; ipol<npol; ipol++)
            {
              float * into = output->get_datptr (ochan, ipol) + heap_val_offset;
              for (uint32_t ival=0; ival<nval_per_packet; ival++)
              {
                into[ival] = static_cast<float>(in[0]) * multiplier;
                in++;
              }
            }
          }
          packet_number++;
        }
      }
    }

    //! flag for whether the custom ASCIIObservation parameters have been configured
    bool configured = false;

    //! number of channels per packet
    uint32_t nchan_per_packet = 0;

    //! number of samples per packet
    uint32_t nsamp_per_packet = 0;

    //! number of samples per relative weight
    uint32_t nsamp_per_weight = 0;

    //! number of packets per heap
    uint32_t npackets_per_heap = 0;

    //! number of bytes between weights+scales packets
    uint32_t weights_packet_stride = 0;

    //! each scale factor (one per packet) is a 32-bit floating point value
    uint32_t scale_nbyte = 4;

    //! each weights value (one per channel) is a 16-bit unsigned integer value
    uint32_t weight_nbyte = 2;

    //! flag that indicates the weights are valid and should be respected
    bool weights_valid = false;

    //! number of values per datum in the input bitseries, 2=complex
    unsigned ndim = 0;

    //! number of polarisations in the input bitseries
    unsigned npol = 0;

    //! number of bits per datum
    unsigned nbit = 0;

    //! total number of channels in the input bitseries
    unsigned nchan = 0;

    //! name of the instrument that generated the bitseries
    std::string machine;

    //! total number of invalid packets encountered
    uint64_t invalid_packets = 0;

    bool device_prepared;

  protected:

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  };

  class SKAParallelUnpacker::Engine : public Reference::Able
  {
  public:

    /**
     * @brief Unpack interface for engines from the input BitSeries to the output TimeSeries
     *
     * @param data input data bitseries
     * @param weights input weights bitseries
     * @param output output timeseries
     * @param nsamp_per_packet number of samples per UDP packet
     * @param nchan_per_packet number of channels per UDP packet
     * @param nsamp_per_weight number samples per relative weight
     * @param nsamp_per_weight number samples per relative weight
     * @param weights_valid flag that indicates the weights are valid and should be respected
     */
    virtual void unpack(const BitSeries * data, const BitSeries * weights, TimeSeries * output, uint32_t nsamp_per_packet, uint32_t nchan_per_packet, uint32_t nsamp_per_weight, bool weights_valid) = 0;

    //! Perform any internal setup
    virtual void setup (SKAParallelUnpacker* user);

  };

} // namespace dsp

#endif // __dsp_Kernel_Formats_ska1_SKAParallelUnpacker
