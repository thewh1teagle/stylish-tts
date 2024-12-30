# Accept a train or val list on standard input and split it into many
# lists based on the duration of the clips.

import soundfile as sf
import re, sys

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
        print("DUPLICATE", fields[0].strip())
    else:
        already[fields[0].strip()] = True
    audio, sample_rate = sf.read("wav/"
                                 + fields[0].strip())
    binkey = length_to_bin(audio.shape[0] // 300)
    if binkey not in time_bins:
        time_bins[binkey] = []
    time_bins[binkey].append((fields[0].strip(), fields[1].strip()))

for key in sorted(time_bins.keys()):
    #print(str(key) + ":", len(time_bins[key]))
    filename = ("train/list-%d.txt" % key)
    with open(filename, "w") as f:
        for item in time_bins[key]:
            text = re.sub(r"([↗↘])", r" \1 ", item[1])
            f.write("%s|%s|0\n" % (item[0], text))
