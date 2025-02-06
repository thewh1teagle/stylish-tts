import sys
import torch
import numpy
import datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
import sentence_transformers
from sentence_transformers.training_args import BatchSamplers


# variant = "prosody"

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

model = SentenceTransformer("microsoft/mpnet-base")
# model = SentenceTransformer("nomic-ai/modernbert-embed-base")
dense = sentence_transformers.models.Dense(
    in_features=model.get_sentence_embedding_dimension(),
    # out_features=128,
    out_features=256,
    bias=False,
    activation_function=torch.nn.Identity(),
)
model.add_module("dense", dense)
model = model.to("cuda")

# DATASETS

exdict = {}


def make_dataset(exemplars, textpath, letter):
    dict = {}
    # exdict[letter] = exemplars
    # keylist = [ letter + str(i) for i in range(exemplars.shape[0])]
    dict["label"] = exemplars.tolist()  # keylist
    with open(textpath) as f:
        lines = []
        for line in f:
            if len(line.strip()) > 0:
                lines.append(line.strip())
        dict["input"] = lines

    return datasets.Dataset.from_dict(dict)


alldata = numpy.load("sentence-data.npz", allow_pickle=False)
train_set = make_dataset(alldata["style_train"], "sentence-train.txt", "t")
# print(train_set[0])
# quit()
val_set = make_dataset(alldata["style_val"], "sentence-val.txt", "v")

# LOSS


class StyleLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        # print(sentence_features)
        prediction = self.model(sentence_features[0])
        gt = labels
        # gt = None
        # if variant == "style":
        #    gt = labels[:, :128]
        # else:
        #    gt = labels[:, 128:]
        return torch.nn.functional.l1_loss(prediction["sentence_embedding"], gt)


loss = StyleLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    # output_dir="sbert-" + variant,
    output_dir="sbert",
    # Optional training parameters:
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
)


trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_set, eval_dataset=val_set, loss=loss
)
trainer.train()

# model.save_pretrained("sbert-" + variant + "/final")
model.save_pretrained("sbert/final")
