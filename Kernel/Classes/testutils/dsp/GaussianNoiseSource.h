//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_GaussianNoiseSource_h
#define __dsp_GaussianNoiseSource_h

#include <dsp/TestSource.h>

namespace dsp::test {

/**
 * @brief Class that imitates a Source of data, useful as a test harness for pipelines.
 *
 */
class GaussianNoiseSource : public TestSource
{
  public:
    /**
     * @brief Construct a new GaussianNoiseSource object
     *
     * @param niterations number of operation iterations to perform.
     */
    GaussianNoiseSource(unsigned niterations = 1);

    /**
     * @brief Destroy the GaussianNoiseSource object
     *
     */
    ~GaussianNoiseSource() = default;

    //! increment the number of iterations performed.
    void operation();

    /**
     * @brief Return a default constructed clone of self.
     *
     * @return Source* clone of this object.
     */
    Source* clone() const;

};

} // namespace dsp

#endif // !defined(__dsp_GaussianNoiseSource_h)
