from pydub import AudioSegment
import random, os, sys, re, pathlib, argparse
import torch
import numpy
import inference
from sentence_transformers import SentenceTransformer

prefix = "a"
sentence_model = SentenceTransformer("all-mpnet-base-v2")


def get_style(audio, model):
    audio.export("temp.wav", format="wav", parameters=["-ar", "24000", "-ac", "1"])
    return model.calculate_audio_style("temp.wav")


def get_embedding(text):
    return sentence_model.encode(text)


def seek_sentence(index, phrases, chapter_length):
    text = ""
    end_index = index
    start = 0
    end = 0
    while index < len(phrases) and phrases[index][2] is None:
        index += 1
    if index < len(phrases):
        count = 0
        start = max(0, phrases[index][0] - 50)
        if index > 0 and phrases[index - 1][1] is not None:
            start = max(phrases[index - 1][1], start)
        end = start
        done = False
        while not done:
            can_lookahead = (
                index < len(phrases) - 1 and phrases[index + 1][2] is not None
            )
            end = min(chapter_length, phrases[index][1] + 50)
            if can_lookahead:
                end = min(phrases[index + 1][0], end)
            else:
                done = True
            text = text + " " + phrases[index][2]
            if "." in text or len(text) > 400:
                done = True
            count = count + 1
            index = index + 1
    end_index = index
    return (end_index, start, end, text)


def main_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument(
        "--model", default="/home/duerig/proj/tts/models/bc-phon-06.pth"
    )
    parser.add_argument(
        "--config", default="/home/duerig/proj/tts/models/bc-phon-06.yml"
    )
    parser.add_argument("--styledir", default="./")
    args = parser.parse_args()
    base = pathlib.Path(args.base)
    model = inference.Model(args.styledir, args.config, args.model)

    chapters = {}
    filename = base / "raw/match-merged.txt"
    with filename.open(mode="r") as f:
        name = ""
        for line in f:
            fields = line.split("|")
            if fields[0] == "chapter":
                name = fields[1].strip()
                chapters[name] = []
            elif fields[0] == "phrase":
                chapters[name].append(
                    (int(fields[1].strip()), int(fields[2].strip()), fields[3].strip())
                )
            else:
                chapters[name].append((None, None, None))

    with torch.no_grad():
        data = {
            "style_train": [],
            "style_val": [],
            "embedding_train": [],
            "embedding_val": [],
        }
        chapter_number = 1
        chapter_total = len(chapters.keys())
        valfile = open("sentence-val.txt", "w")
        trainfile = open("sentence-train.txt", "w")
        for key in chapters.keys():
            print("(%d/%d) Processing %s\n" % (chapter_number, chapter_total, key))
            chapter_audio = AudioSegment.from_mp3(str(base / key))
            chapter_length = len(chapter_audio)
            phrases = chapters[key]
            index = 0
            while index < len(phrases):
                (index, begin, end, text) = seek_sentence(
                    index, phrases, chapter_length
                )
                if (
                    len(text) > 0
                    and len(text) < 500
                    and len(text.split(".")) == 2
                    and text.strip()[-1] == "."
                    and end - begin > 1000
                    and end - begin < 30000
                ):
                    embedding = get_embedding(text)
                    style = get_style(chapter_audio[begin:end], model)
                    style = style.squeeze().cpu().numpy()
                    if random.random() < 0.05:
                        data["style_val"].append(style)
                        data["embedding_val"].append(embedding)
                        valfile.write(text + "\n")
                    else:
                        data["style_train"].append(style)
                        data["embedding_train"].append(embedding)
                        trainfile.write(text + "\n")
                    # print(embedding.shape, style.shape)
                    sys.stdout.write(".")
                    sys.stdout.flush()
            chapter_number += 1

        for key in data.keys():
            data[key] = numpy.stack(data[key])
            print(data[key].shape)

        numpy.savez_compressed("sentence-data.npz", allow_pickle=False, **data)
        valfile.close()
        trainfile.close()


main_method()
