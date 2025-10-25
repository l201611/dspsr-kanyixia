# How `dspsr` ingests Mark5B recordings

## 1. Discovering the dataset
- `Mark5bFile::is_valid` looks for a sibling header named `<data>.hdr` and
  refuses the file if it is missing. After reading the first kilobyte it
  insists that the ASCII header supplies a `FORMAT` keyword, otherwise the
  Mark5B backend is not considered a match.【F:Kernel/Formats/mark5b/Mark5bFile.C†L33-L55】

## 2. Opening the Mark5B stream
- Once the header is found, `Mark5bFile::open_file` loads it and again
  requires `FORMAT`. The value is forwarded to `mark5access` via
  `new_mark5_format_generic_from_string`, so it must follow the
  `Format-Mbps-nChannels-nBits` pattern understood by that library
  (for example `Mark5B-1024-16-2`).【F:Kernel/Formats/mark5b/Mark5bFile.C†L57-L114】
- The same routine constructs a `mark5_stream` by combining the descriptor
  string with the raw data file. `Input::resolution` is then aligned with the
  stream's `samplegranularity` to ensure `dspsr` reads whole Mark5B frames at a
  time.【F:Kernel/Formats/mark5b/Mark5bFile.C†L115-L119】

## 3. Timing metadata
- `open_file` accepts either `REFMJD` (added to the base MJD supplied by the
  stream) or a direct `MJD` keyword to seed the observation start time, using
  the seconds counter extracted from the Mark5B frames.【F:Kernel/Formats/mark5b/Mark5bFile.C†L120-L134】

## 4. Required observational keywords
- The ASCII header must also provide `TELESCOPE`, `SOURCE`, centre `FREQ`,
  and total `BW`; optional `RA`/`DEC` strings are translated to sky
  coordinates when present.【F:Kernel/Formats/mark5b/Mark5bFile.C†L139-L216】

## 5. Deriving stream layout
- After `mark5access` reports the stream parameters, `Mark5bFile` computes the
  number of polarisations by comparing the Mbps rate, bit depth, and recorded
  bandwidth. It then stores the corresponding channel count, bit depth,
  sample rate, and flags the data as Nyquist-sampled Mark5B voltage.
  Finally the machine name is stamped as `Mark5b` so that the correct unpacker
  can claim the stream.【F:Kernel/Formats/mark5b/Mark5bFile.C†L217-L262】

## 6. Unpacking
- The Mark5B backend installs a dedicated unpacker. Once the file handler sets
  `machine="Mark5b"`, `Mark5bUnpacker::matches` claims the observation and asks
  the `mark5access` library to decode each gulp. Channels are presented to
  `dspsr` in the familiar frequency-major, polarisation-minor order by
  arranging the output pointers before the decode call.【F:Kernel/Formats/mark5b/Mark5bUnpacker.C†L20-L60】

## 7. Registration
- Both the file handler and the unpacker are automatically registered when the
  build is compiled with `HAVE_mark5b`, so simply pointing `dspsr` at a Mark5B
  header/data pair activates this pipeline.【F:Kernel/Formats/File_registry.C†L129-L137】【F:Kernel/Formats/Unpacker_registry.C†L180-L190】

In summary, a Mark5B observation consists of a raw file plus a `.hdr`
metadata block describing the stream format. `dspsr` uses the ASCII header to
bootstrap `mark5access`, derives the timing and layout from the library's
report, and hands decoding duties to the Mark5B-specific unpacker so the rest
of the pulsar processing chain sees standard, channel-ordered voltage samples.
