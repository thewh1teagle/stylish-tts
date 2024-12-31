# Accept a train or val list on standard input and split it into many
# lists based on the duration of the clips.
#
# --wav is the directory of the wav files to check
# --out is the output directory where a bunch of segment lists will be spawned

import soundfile as sf
import argparse, pathlib, re, sys

parser = argparse.ArgumentParser()
parser.add_argument("--wav", default="wav/")
parser.add_argument("--out", default="out/")
args = parser.parse_args()

wavdir = pathlib.Path(args.wav)
outdir = pathlib.Path(args.out)
if not outdir.exists():
    outdir.mkdir(parents=True)

already = {}
time_bins = {}

def length_to_bin(length):
    if length < 100:
        bin = 0
    else:
        bin = (length - 80) // 20
    return bin

for line in sys.stdin:
    fields = line.split("|")
    if fields[0].strip() in already:
        sys.stderr.write("DUPLICATE " + fields[0].strip())
    else:
        already[fields[0].strip()] = True
    audio, sample_rate = sf.read(str(wavdir / fields[0].strip()))
    binkey = length_to_bin(audio.shape[0] // 300)
    if binkey not in time_bins:
        time_bins[binkey] = []
    time_bins[binkey].append((fields[0].strip(), fields[1].strip(), fields[2].strip()))

for key in sorted(time_bins.keys()):
    #print(str(key) + ":", len(time_bins[key]))
    filename = outdir / ("list-%d.txt" % key)
    with filename.open("w") as f:
        for item in time_bins[key]:
            f.write("%s|%s|%s\n" % (item[0], item[1], item[2]))
