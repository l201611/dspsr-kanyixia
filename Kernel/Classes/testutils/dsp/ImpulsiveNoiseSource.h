//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_ImpulsiveNoiseSource_h
#define __dsp_ImpulsiveNoiseSource_h

#include <dsp/TestSource.h>

namespace dsp::test {

/**
 * @brief Class that creates a repeating impulse.
 *
 * This class doesn't add noise but is a pure signal that has a duration of being
 * on, a height/strength of the signal, and a period of the signal.  This is effectively
 * a square wave signal with a duty cycle of duration/period.
 *
 * If Gaussian noise is also needed, use a @see SumSource along with a @see GaussianNoiseSource
 * to combine an impulsive noise signal within Gaussian noise.
 *
 * The default period is 1.0 second, with a duration of 10 milliseconds (i.e. a duty cycle of 1%)
 * and a height scale of 1.0.
 */
class ImpulsiveNoiseSource : public TestSource
{
  public:
    /**
     * @brief Construct a new ImpulsiveNoiseSource object
     *
     * @param niterations number of operation iterations to perform.
     */
    ImpulsiveNoiseSource(unsigned niterations = 1);

    /**
     * @brief Destroy the ImpulsiveNoiseSource object
     *
     */
    virtual ~ImpulsiveNoiseSource() = default;

    //! increment the number of iterations performed.
    void operation();

    /**
     * @brief Return a default constructed clone of self.
     *
     * @return Source* clone of this object.
     */
    dsp::Source* clone() const;

    /**
     * @brief set the period, in seconds, of the impulsive noise signal.
     *
     * @param _period the period, in seconds, of the impulsive noise signal.
     * @throws Error if period is less than or equal to zero.
     */
    void set_period(double _period);

    /**
     * @brief get the period of the impulsive noise signal, in seconds.
     *
     * @returns the period of the impulsive noise signal, in seconds.
     */
    double get_period() const { return period; }

    /**
     * @brief set the duration of the impulsive noise signal, in seconds.
     *
     * The on signal duty cycle is equivalent to the duration / period.
     *
     * This value needs to be greater or equal to zero.  If the value is
     * zero then there is no impulses.
     *
     * @param _duration the duration of the impulsive noise signal, in seconds.
     * @throws Error if the duration is negative.
     */
    void set_impulse_duration(double _impulse_duration);

    /**
     * @brief get the duration of the impulsive noise signal, in seconds.
     *
     * @returns the duration of the impulsive noise signal, in seconds.
     */
    double get_impulse_duration() const { return impulse_duration; }

    /**
     * @brief set the height of the impulsive signal.
     *
     * The units of this value depends on the units of the Timeseries it is representing.
     *
     * @param _height the height of the impulsive signal.
     */
    void set_height(double _height) { height = _height; }

    /**
     * @brief get the height of the impulsive signal.
     *
     * The units of this value depends on the units of the Timeseries it is representing.
     *
     * @returns the height of the impulsive signal.
     */
    float get_height() const { return height; }

    /**
     * @brief set the phase offset of when the impulse noise should start.
     *
     * @param _phase_offset the phase offset to use.
     */
    void set_phase_offset(float _phase_offset) { phase_offset = fmod(_phase_offset, 1.0); }

    /**
     * @brief get the phase offset of when the impulse noise should start.
     */
    float get_phase_offset() const { return phase_offset; }

  private:
    //! Generate FPT ordered data
    void generate_fpt();

    //! Generate TFP ordered data
    void generate_tfp();

    //! The period of the impulsive noise signal, in seconds.
    double period{1.0};

    //! The duration, in seconds, of the how long the impulse lasts for.
    double impulse_duration{0.01};

    //! The strength of the impulsive noise. Units are that of the signal itself (e.g. volts for complex voltages).
    float height{1.0};

    //! The phase offset of when the on signal occurs.
    float phase_offset{0.5};
};

} // namespace dsp::test

#endif // !defined(__dsp_ImpulsiveNoiseSource_h)
