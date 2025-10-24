//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/WeightedTimeSeries.h

#ifndef __WeightedTimeSeries_h
#define __WeightedTimeSeries_h

#include "dsp/TimeSeries.h"

namespace dsp {

  //! Container of weighted time-major order floating point data.
  /* The WeightedTimeSeries class contains floating point data that
     may be flagged as bad in the time domain. */
  class WeightedTimeSeries : public TimeSeries {

  public:

    //! Default constructor
    WeightedTimeSeries ();

    //! Default destructor
    ~WeightedTimeSeries ();

    //! Copy constructor
    WeightedTimeSeries (const WeightedTimeSeries&);

    //! Assignment operator
    const WeightedTimeSeries& operator = (const WeightedTimeSeries&);

    //! Return a copy-constructed clone (calls new)
    WeightedTimeSeries* clone() const override;

    //! Returns a default-constructed WeightedTimeSeries (calls new)
    WeightedTimeSeries* null_clone() const override;

    //! Add each value in data to this
    void add (const TimeSeries*) override;

    //! Add each value in data to this
    void add (const WeightedTimeSeries*);

    //! Set the number of time samples per weight
    /*! Set ndat_per_weight to zero to effect no weighting of data */
    void set_ndat_per_weight (unsigned ndat_per_weight);

    //! Get the number of time samples per weight
    unsigned get_ndat_per_weight () const { return ndat_per_weight; }

    //! Set the number of polarizations with independent weights
    void set_npol_weight (unsigned npol_weight);

    //! Get the number of polarizations with independent weights
    unsigned get_npol_weight () const { return npol_weight; }

    //! Set the number of frequency channels with independent weights
    void set_nchan_weight (unsigned nchan_weight);

    //! Get the number of frequency channels with independent weights
    unsigned get_nchan_weight () const { return nchan_weight; }

    //! Set the reserve kludge factor
    void set_reserve_kludge_factor (unsigned);

    //! Copy the configuration of another WeightedTimeSeries instance
    void copy_configuration (const Observation* copy) override;

    //! Copy the dimensions of another TimeSeries instance
    void copy_dimensions (const Observation* copy) override;

    //! Copy the data of another WeightedTimeSeries instance
    void copy_data (const TimeSeries* data, uint64_t idat_start = 0, uint64_t ndat = 0) override;

    //! Allocate the space required to store nsamples time samples.
    void resize (uint64_t nsamples) override;

    //! Returns the number of uint16_t between consecutive blocks of weights
    uint64_t get_weights_stride () const { return weight_subsize; }

    //! Offset the base pointer by offset time samples
    void seek (int64_t offset) override;

    //! Set all values to zero
    void zero () override;

    //! Maybe copy the weights from copy
    void copy_weights (const Observation* copy);

    //! For each zero weight, sets all weights to zero
    void mask_weights ();

    //! Check that each floating point value is zeroed if weight is zero
    void check_weights ();

    //! Set all weights to one
    void neutral_weights ();

    //! Get the number of weights
    uint64_t get_nweights () const;

    //! Get the number of weights required for a given number of samples
    uint64_t get_nweights (uint64_t nsample) const;

    //! Set the offset of the first time sample in the current weight array
    void set_weight_idat (uint64_t weight_idat);

    //! Get the offset into the current weight of the first time sample
    uint64_t get_weight_idat () const { return weight_idat; }

    //! Get the number of zero weights in the ichan == ipol == 0 array
    uint64_t get_nzero () const;

    //! Get the weights array for the specfied polarization and frequency
    uint16_t* get_weights (unsigned ichan=0, unsigned ipol=0);

    //! Get the weights array for the specfied polarization and frequency
    const uint16_t* get_weights (unsigned ichan=0, unsigned ipol=0) const;

    //! Flag all weights in corrupted transforms
    void convolve_weights (unsigned nfft, unsigned nkeep);

    //! Scrunch the weights
    void scrunch_weights (unsigned nscrunch);

    //! Return the internal memory base address
    uint16_t* internal_get_weights_buffer() { return base; }
    const uint16_t* internal_get_weights_buffer() const { return base; }

    //! Return the internal memory size in bytes
    uint64_t internal_get_weights_size() const { return weight_size * sizeof(uint16_t); }

    //! Return the internal memory sub-division size in bytes
    uint64_t internal_get_weights_subsize() const { return weight_subsize * sizeof(uint16_t); }

    //! Set the weights memory manager
    /*! Weights may be stored on a device other than the one used to store floating-point data. */
    void set_weights_memory (Memory*);
    Memory* get_weights_memory ();
    const Memory* get_weights_memory () const;

  protected:

    //! Number of polarizations with independent weights
    unsigned npol_weight = 1;

    //! Number of frequency channels with independent weights
    unsigned nchan_weight = 1;

    //! The number of time samples per weight
    unsigned ndat_per_weight = 0;

    //! The reserve kludge factor is required by the Filterbank
    unsigned reserve_kludge_factor = 1;

    //! TimeSeries::internal_match(DataSeries) override calls virtual internal_match(TimeSeries) if appropriate
    using DataSeries::internal_match;

    //! Match the internal memory layout of another TimeSeries
    void internal_match (const TimeSeries*) override;

    //! Match the internal weights memory layout of another WeightedTimeSeries
    void internal_match_weights (const WeightedTimeSeries*);

    //! Copy selected weights from input
    void copy_weights (const WeightedTimeSeries* input, uint64_t idat_start = 0, uint64_t copy_ndat = 0);

    //! Copy all weights from input
    void copy_all_weights (const WeightedTimeSeries* input);

    //! Resize the weights array
    void resize_weights (uint64_t nsamples);

    //! Get the number of weights possible given allocated space
    uint64_t have_nweights () const;

    void prepend_checks (const TimeSeries*, uint64_t pre_ndat);

  private:

    //! The base of the weights buffer
    uint16_t* base = nullptr;

    //! The pointer to the current start of weights buffer (can be seeked)
    uint16_t* weights = nullptr;

    //! The index into the first weight of the first time sample
    uint64_t weight_idat = 0;

    //! The size of the buffer
    uint64_t weight_size = 0;

    //! The size of each division of the buffer
    uint64_t weight_subsize = 0;

    //! By default, weights memory are on host
    bool weights_on_host = true;

    //! The weights memory manager
    Reference::To<Memory> weights_memory;
  };

}

#endif

