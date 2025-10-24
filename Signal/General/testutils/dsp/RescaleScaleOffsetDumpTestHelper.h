/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Rescale.h"
#include "ReferenceAble.h"

#include <vector>
#include <string>

#ifndef __dsp_RescaleScaleOffsetDumpTestHelper_h
#define __dsp_RescaleScaleOffsetDumpTestHelper_h

namespace dsp::test {

/**
 * @brief a test utility class that helps verifying scales and offset file.
 *
 * This class can be used as a callback handler for when Rescale has update the scales and offsets
 * and it can then use the data it captures to then verify an output.
 */
class RescaleScaleOffsetDumpTestHelper : public Reference::Able {

  public:
    /**
     * @brief verify a scales and offsets DADA file based on the data captured during a test.
     *
     * @param filename the name of the file to verify.
     */
    void assert_file(std::string filename);

    /**
     * @brief the callback method used when Rescale has updated the scales and offsets.
     *
     * @param rescale the pointer to the instance of Rescale that just performed and update of the scales and offsets.
     */
    void rescale_update(dsp::Rescale::update_record record);

    /**
     * @brief get the number of updates (i.e. calls to rescale_update).
     *
     * This is a proxy for the number of times the rescale_update has been called.  It allows
     * for assertions of when using constant rescale versus varying rescale during a pipeline.
     */
    uint64_t get_num_updates() { return num_updates; };

    /**
     * @brief get the scales and offset update records.
     */
    const std::vector<Rescale::update_record>& get_records() { return records; }

  private:
    //! number of updates that have been performed
    uint64_t num_updates{0};

    //! captured data (capture of the scale and offsets)
    std::vector<Rescale::update_record> records;
};

}

#endif // __dsp_RescaleScaleOffsetDumpTestHelper_h
