//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Observation_h
#define __Observation_h

#include "dsp/dsp.h"

#include "OwnStream.h"
#include "sky_coord.h"
#include "Types.h"
#include "MJD.h"
#include "Rational.h"
#include "dsp/FIRFilter.h"

#include <vector>
#include <cstdint>

// forward declaration of text interface
namespace TextInterface
{
  class Parser;
};

namespace dsp
{
  //! Stores information about digital, band-limited, time-varying signals
  class Observation : public OwnStream
  {

  public:

    //! Verbosity flag
    static bool verbose;

    //! Null constructor
    Observation ();

    //! Copy constructor
    Observation (const Observation&);

    //! Copy from pointer constructor
    Observation (const Observation*);

    //! Assignment operator
    const Observation& operator = (const Observation&);

    //! Virtual destructor (see Effective C++ Item 14)
    virtual ~Observation();

    //! Same as operator= but takes a pointer
    virtual void copy (const Observation* obs) { operator=( *obs ); }

    //! Cloner (calls new)
    virtual Observation* clone() const;

    //! Set the type of receiver feeds
    virtual void set_basis (Signal::Basis _basis) { basis = _basis; }
    //! Return the type of receiver feeds
    Signal::Basis get_basis () const { return basis; }

    //! Set the state of the signal
    virtual void set_state (Signal::State _state);
    //! Return the state of the signal
    Signal::State get_state () const { return state; }

    //! Set the source type
    virtual void set_type (Signal::Source _type) { type = _type; }
    //! Return the source type
    Signal::Source get_type () const { return type; }

    //! Set the dimension of each datum
    virtual void set_ndim (unsigned _ndim) { ndim = _ndim; }
     //! Return the dimension of each datum
    unsigned get_ndim () const { return ndim; }

     //! Set the number of channels into which the band is divided
    virtual void set_nchan (unsigned _nchan) { nchan = _nchan; }
    //! Return the number of channels into which the band is divided
    unsigned get_nchan () const { return nchan; }

    //! Set the number of polarizations
    virtual void set_npol (unsigned _npol) { npol = _npol; }
    //! Return the number of polarizations
    unsigned get_npol () const { return npol; }

    //! Set the number of bits per value
    virtual void set_nbit (unsigned _nbit) { nbit = _nbit; }
    //! Return the number of bits per value
    unsigned get_nbit () const { return nbit; }

    //! Return the number of bits in a single time sample (over all channels, pols, and dimensions)
    uint64_t bits_per_sample () const;

    //! Set the number of time samples in container
    /*! Note that one time sample may be complex and/or vector in
      nature.  For instance, the in-phase and quadrature components of
      two orthogonal polarizations, though represented by four
      independent numbers, still represent one time sample. */
    virtual void set_ndat (uint64_t _ndat) { ndat = _ndat; }
    //! Return the number of time samples in container
    uint64_t get_ndat () const { return ndat; }

    //! Set the telescope name
    virtual void set_telescope (const std::string& name) { telescope = name; }
    //! Return the telescope name
    std::string get_telescope () const { return telescope; }

    //! Set the receiver name
    virtual void set_receiver (const std::string& name) { receiver = name; }
    //! Return the receiver name
    std::string get_receiver () const { return receiver; }

    //! Set the source name
    virtual void set_source (const std::string& name) { source = name; }
    //! Return the source name
    std::string get_source () const { return source; }

    //! Set the coordinates of the source
    virtual void set_coordinates (sky_coord _coordinates)
    { coordinates=_coordinates; }
    //! Return the coordinates of the source
    sky_coord get_coordinates () const
    { return coordinates; }

    //! Set the centre frequency of the band-limited signal in MHz
    virtual void set_centre_frequency (double cf) { centre_frequency = cf; }
    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const{ return centre_frequency; }

    //! Returns the centre frequency of the specified channel in MHz
    double get_centre_frequency (unsigned ichan) const;
    //! Returns the centre frequency of the reference channel in MHz
    double get_base_frequency () const;

    //! Returns the unswapped channel index of the specified channel
    unsigned get_unswapped_ichan (unsigned ichan) const;

    //! Set the bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    virtual void set_bandwidth (double _bandwidth) { bandwidth = _bandwidth; }
    //! Return the bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double get_bandwidth () const { return bandwidth; }

    //! Set the start time of the leading edge of the first time sample
    virtual void set_start_time (MJD _start_time) { start_time = _start_time; }
    //! Return the start time of the leading edge of the first time sample
    MJD get_start_time () const { return start_time; }

    //! Set the sampling rate (time samples per second in Hz)
    /*! Determines the sampling interval. */
    virtual void set_rate (double rate);
    //! Return the sampling rate (time samples per second in Hz)
    double get_rate () const { return rate; }

    //! Set the sampling interval (in seconds)
    /*! Alternative method of setting the rate. */
    void set_interval (double interval);
    //! Return the sampling interval (in seconds)
    double get_interval () const;
 
    //! Set the amount by which data has been scaled
    virtual void set_scale (double _scale) { scale = _scale; }
    //! Return the amount by which data has been scaled
    double get_scale () const { return scale; }

    //! Set true if frequency channels are out of order (band swappped)
    virtual void set_swap (bool _swap) { swap = _swap; }
    //! Return true if frequency channels are out of order (band swappped)
    bool get_swap () const { return swap; }

    //! Set the number of sub-bands that must be band swapped
    virtual void set_nsub_swap (unsigned _nsub) { nsub_swap = _nsub; }
    //! Return the number of sub-bands that must be band swapped
    unsigned get_nsub_swap () const { return nsub_swap; }

    //! Set to 1 if the spectrum in each channel has dual-sideband frequency order
    /*! Set to N if the spectrum is also composed of N dual-sideband sub-bands.
        Set to -N if the spectrum is only composed of N dual-sideband sub-bands. */
    virtual void set_dual_sideband (int _dual);
    //! Return the dual-sideband
    int get_dual_sideband () const;

    //! Set true if centre channel is centred on centre frequency
    /*! This flag is currently experimental */
    virtual void set_dc_centred (bool _centred) { dc_centred = _centred; }
    bool get_dc_centred () const { return dc_centred; }

    //! Set the observation identifier
    virtual void set_identifier (const std::string& _id) { identifier = _id; }
    //! Return the observation identifier
    std::string get_identifier () const { return identifier; }

    //! Set the instrument used to record signal
    virtual void set_machine (const std::string& _m) { machine = _m; }
    //! Return the instrument used to record signal
    std::string get_machine () const { return machine; }

    //! Set the format used to record signal
    virtual void set_format (const std::string& _f) { format = _f; }
    //! Return the format used to record signal
    std::string get_format () const { return format; }

    //! Set the pulsar dispersion mesure
    virtual void set_dispersion_measure (double dm)
    { dispersion_measure = dm; }
    //! Returns the pulsar dispersion measure
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the pulsar rotation mesure
    virtual void set_rotation_measure (double rm)
    { rotation_measure = rm; }
    //! Returns the pulsar rotation measure
    double get_rotation_measure () const
    { return rotation_measure; }

    //! Set the observation mode
    virtual void set_mode (const std::string& _mode) { mode = _mode; }
    //! Return the observation mode
    std::string get_mode () const { return mode; }

    //! Set the calibrator frequency
    virtual void set_calfreq (double _calfreq) {calfreq = _calfreq;}
    //! get the calibrator frequency
    double get_calfreq() const {return calfreq;}

    //! Set the oversampling factor
    virtual void set_oversampling_factor (const Rational& _osf)
    { oversampling_factor = _osf; }

    //! Get the oversampling factor
    const Rational& get_oversampling_factor () const
    { return oversampling_factor; }

    virtual void set_deripple (const std::vector<dsp::FIRFilter>& _deripple)
    { deripple = _deripple; }

    const std::vector<dsp::FIRFilter>& get_deripple () const { return deripple; }

    // get the number of derippling stages,
    // or the number of upsteam layers of channelization
    unsigned get_deripple_stages () const { return deripple.size(); }

    //! set/get the DC PFB channel flag (for inverse filterbank)
    virtual void set_pfb_dc_chan (const bool _pfb_dc_chan) { pfb_dc_chan = _pfb_dc_chan; }

    bool get_pfb_dc_chan () const { return pfb_dc_chan; }

    //! set/get the number of channels output by the PFB per input channel
    virtual void set_pfb_nchan (unsigned nchan) { pfb_nchan = nchan; }
    unsigned get_pfb_nchan () const { return pfb_nchan; }

    //! Change the state and correct other attributes accordingly
    virtual void change_state (Signal::State new_state);

    //! Return true if the state of the Observation is valid
    bool state_is_valid (std::string& reason) const;

    //! Returns true if state is Detected, Coherence, or Stokes
    bool get_detected () const;

    //! Change the start time by the number of time samples specified
    void change_start_time (int64_t _ndat);

    //! Return the end time of the trailing edge of the last time sample
    // Returns correct answer if ndat=rate=0 and avoids division by zero
    virtual MJD get_end_time () const;

    //! Return the nearest time sample corresponding to the specified MJD
    uint64_t get_idat (const MJD& mjd);

    //! Multiply scale by factor
    void rescale (double factor) { scale *= factor; }

    //! Return the size in bytes of nsamples time samples
    virtual uint64_t get_nbytes (uint64_t nsamples) const;

    //! Return the size in bytes of ndat time samples
    uint64_t get_nbytes () const
      { return get_nbytes (get_ndat()); }

    uint64_t verbose_nbytes (uint64_t nsamples) const;

    //! Return the size in bytes of one time sample
    float get_nbyte () const;

    //! Return the number of samples in nbytes bytes
    virtual uint64_t get_nsamples (uint64_t nbytes) const;

    //! Copy the dimensions of another observation
    virtual void copy_dimensions (const Observation*);

    //! Copy the transient attributes of another observation
    virtual void copy_transient_attributes (const Observation*);

    //! Returns true if the signal may be integrated
    bool combinable (const Observation& obs) const;

    //! Returns the reason if combinable returns false
    std::string get_reason () { return reason; }

    //! Return true if the first sample of next follows the last sample of this
    bool contiguous (const Observation& next) const;

    //! Set all attributes to null default
    void init ();

    //! Defines the bijection between number of bytes and number of samples
    class NbyteNsamplePolicy;

    //! Set the policy that defines the bijection between number of bytes and number of samples
    void set_nbyte_nsample_policy (NbyteNsamplePolicy*);

    //! Get the policy that defines the bijection between number of bytes and number of samples
    const NbyteNsamplePolicy* get_nbyte_nsample_policy () const;

    //! Textual interface to Observation attributes
    class Interface;

    //! Return a text interface that can be used to access this instance
    virtual TextInterface::Parser* get_interface ();

  protected:

    //! Telescope name
    std::string telescope;

    //! Receiver name
    std::string receiver;

    //! Source name.  If a pulsar, should be J2000
    std::string source;

    //! Centre frequency of band-limited signal in MHz
    double centre_frequency = 0.0;

    //! Bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double bandwidth = 0.0;

    //! Type of signal source (Pulsar, CAL, etc.)
    Signal::Source type;

    //! State of the signal (Full Stokes, Total Intensity, etc.)
    Signal::State state;

    //! Type of receiver feeds (Linear or Circular)
    Signal::Basis basis;

    //! Time samples per second in Hz
    double rate = 0.0;

    //! Start time of the leading edge of the first time sample
    MJD start_time;

    //! Observation identifier
    std::string identifier;

    //! Observation mode
    std::string mode;

    //! Instrument used to record signal
    std::string machine;

    //! Data format used to record signal
    std::string format;

    //! Coordinates of the source
    sky_coord coordinates;

    //! The dispersion measure to be archived
    double dispersion_measure = 0.0;

    //! The rotation measure to be archived
    double rotation_measure = 0.0;

    //! Require equal sources in combinable
    bool require_equal_sources = true;

    //! Require equal rates in combinable
    bool require_equal_rates = true;

    //! Amount by which data has been scaled
    double scale = 1.0;

    //! Flag set when frequency channels are out of order (band swappped)
    bool swap = false;

    //! The number of sub-bands that must be band swapped
    unsigned nsub_swap = 0;

    //! Flag set when centre channel is centred on centre frequency
    bool dc_centred = false;

    //! oversampling factor
    Rational oversampling_factor;

    //! information about deripple correction
    std::vector<FIRFilter> deripple;

    //! Do we have the DC, or zeroth PFB channel?
    bool pfb_dc_chan = true;

    //! Number of channels output by PFB for a single input time series
    /*! If two stages of analysis filter bank operations are performed,
        pfb_nchan should be the number of fine channels per input coarse channel.

        If only the fine channels that span the critically-sampled portion
        of each coarse channel are kept, then pfb_nchan should be the number
        of fine channels that are kept. */
    unsigned pfb_nchan = 0;

  private:

    /////////////////////////////////////////////////////////////////////
    //
    // Private variables should be accessed only using set/get at all times!
    //

    //! Number of time samples in container
    /*! This is private so that classes that inherit from Observation
      and have nbit%8 != 0 can enforce resizes/set_ndat's so that
      ndat*ndim is always an integer number of bytes */
    uint64_t ndat = 0;

    //! Dimension of each datum
    /*! This is private so that classes that inherit from Observation
      that have nbit%8 != 0 can enforce set_ndim's so that ndat*ndim
      is always an integer number of bytes */
    unsigned ndim = 1;

    //! Number of frequency channels across bandwidth
    unsigned nchan = 1;

    //! Number of polarizations
    unsigned npol = 1;

    //! Number of bits per value
    unsigned nbit = 0;

    //! The calfrequency
    double calfreq = 0.0;

    //! Spectrum of each channel is in dual-sideband oder
    /*!
      0  : -Nyq -> DC -> +Nyq
      1  : DC -> +Nyq, -Nyq -> DC-1
    */
    int dual_sideband = 0;

    bool dual_sideband_set = false;

    //! Reason when combinable fails
    mutable std::string reason;

    //! Policy that defines the bijection between number of bytes and number of samples
    Reference::To<NbyteNsamplePolicy> nbyte_nsample_policy;
  };

  class Observation::NbyteNsamplePolicy : public Reference::Able
  {
  public:

    Observation* obs = nullptr;

    //! Return a new instance of self
    virtual NbyteNsamplePolicy* clone () const;

    //! Return the size in bytes of nsamples time samples
    virtual uint64_t get_nbytes (uint64_t nsamples) const;

    //! Return the number of samples in nbytes bytes
    virtual uint64_t get_nsamples (uint64_t nbytes) const;

    //! Return the number of bits in a single time sample
    uint64_t bits_per_sample () const;
  };

}

#ifdef ACTIVATE_MPI

#include <mpi.h>

//! Return the size required to mpiPack an Observation
int mpiPack_size (const dsp::Observation&, MPI_Comm comm, int* size);

//! Pack an Observation into outbuf
int mpiPack (const dsp::Observation&,
	     void* outbuf, int outcount, int* position, MPI_Comm comm);

//! Unpack an Observation from inbuf
int mpiUnpack (void* inbuf, int insize, int* position,
	       dsp::Observation*, MPI_Comm comm);

#endif

#endif // ! __Observation_h
