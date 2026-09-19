# Selected measured waveforms

Four MATLAB v5 acquisitions are included without numeric modification. Each
contains a float32 column vector of 204,800 values: 256 consecutive frequency
symbols, 800 samples per symbol. Use variable `Data` for the three Data files
and `Dict` for the dictionary file. SHA-256 checksums are in `manifest.json`.

| File suffix | Role in the public example |
|---|---|
| `70MHz_50deg_1.mat` (Data) | sender calibration, repeat 1 |
| `70MHz_50deg_2.mat` (Data) | same nominal state, independent repeat 2 |
| `70MHz_48deg_2.mat` (Data) | different nominal state, repeat 2 |
| `70MHz_50deg_1.mat` (Dict) | dictionary for the separate runtime example |

The acquisition filenames indicate 2026-05-08 and 70-MHz frequency spacing.
The physical link is **20 km of single-mode fiber**, as confirmed by the
authors. The released filenames have been corrected to use `20km`. This is a
filename/metadata correction only; the measurement arrays and file bytes are
unchanged, and the SHA-256 checksums remain valid.
Absolute optical frequency, sampling rate and full instrument settings should
be obtained from the authors before treating these as complete metrology data.

The waveform extractor is the same Gaussian-fit implementation found in the
local and server snapshots. The original timing script uses a different,
calibrated fixed-peak extraction path; the two are not interchangeable timing
benchmarks.

All included state responses are public test fixtures. This small sample
does not provide every state or power setting used in the manuscript.

