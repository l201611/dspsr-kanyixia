# VDIF `.hdr` Requirements in dspsr

This note summarises where `dspsr` enforces requirements on the ASCII sidecar
header that accompanies VDIF recordings.

## Format detection

* `dsp::VDIFFile::is_valid` (`Kernel/Formats/vdif/VDIFFile.C`) opens the header
  and searches for `INSTRUMENT = VDIF`. Any other value causes the probe to
  fail before the VDIF backend is selected.

## Datafile lookup

* When the VDIF backend opens the header, it reads the `DATAFILE` keyword to
  locate the raw VDIF payload. If the keyword is missing, it strips a `.hdr`
  suffix from the metadata filename as a fallback and raises an error when that
  fails.

## Relaxed metadata requirements

* The loader populates an `ASCIIObservation` instance with the header contents
  but explicitly marks timing/quantisation keywords as optional. This means the
  header may omit `UTC_START`, `OBS_OFFSET`, `NBIT`, `NDIM`, and `TSAMP` because
  the VDIF frame headers themselves contain the authoritative values.

## Observation metadata handling

* `ASCIIObservation::load` handles the remaining keywords shared with other
  DSPSR formats. For example, it records the `INSTRUMENT` string as the data
  source name and the standard positional and observation settings when they are
  present.

These routines, taken together, define the minimal ASCII header required for a
VDIF dataset: it must identify `INSTRUMENT = VDIF`, provide either a `DATAFILE`
entry or a matching `.hdr` filename, and can omit timing fields that would be
redundant with the VDIF packet contents.
