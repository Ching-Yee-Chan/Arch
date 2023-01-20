from transformers import AutoModel, AutoFeatureExtractor
import torch
import numpy as np
import soundfile as sf

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset
from arch_eval import ESC50 
from arch_eval import FMASmall
from arch_eval import RAVDESS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/wav2vec2-large-960h')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--tsv_logging_file', type=str, default='results/hf_models.tsv')
args = parser.parse_args()

print("------------------------------------")
print(f"Evaluating model: {args.model}")
print("------------------------------------")

'''
************************************************************************************************
*                                       Setting parameters                                     *
************************************************************************************************
'''

# Load model
audio_model = AutoModel.from_pretrained(args.model)
feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
audio_model = audio_model.to(args.device)
model_parameters = sum(p.numel() for p in audio_model.parameters())
tsv_lines = [] 

'''
************************************************************************************************
*                                         Model Wrapping                                       *
************************************************************************************************
'''

# implement a child class of Model
class Wav2Vec2ModelWrapper(Model):
    def __init__(self, model, feature_extractor, device, max_length):
        super().__init__(model)
        self.model = model
        # the model must not be trained
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_length = max_length

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).input_values
        inputs = inputs.to(self.device)
        token_embeddings = self.model(inputs).last_hidden_state
        return token_embeddings.mean(dim=1).squeeze()


    def get_token_embeddings(self, audio: np.ndarray, **kwargs):

        # TODO: manage long audio files in streaming mode (split in chunks)
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).input_values
        inputs = inputs.to(self.device)
        token_embeddings = self.model(inputs).last_hidden_state
        return token_embeddings.squeeze()

    def get_classification_embedding_size(self):
        return self.model.config.hidden_size

    def get_token_embedding_size(self):
        return self.model.config.hidden_size

    def get_sampling_rate(self):
        return self.feature_extractor.sampling_rate

    def get_embedding_layer(self):
        # return the size of the embedding layer
        return self.model.config.hidden_size



'''
************************************************************************************************
*                                          ESC50                                               *
************************************************************************************************
'''
model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=5*16_000)

ESC50_DATASET_PATH = "/data1/mlaquatra/datasets/audio_datasets/esc50/"
evaluator_esc50 = ESC50(path=ESC50_DATASET_PATH, verbose=args.verbose)
res_esc50 = evaluator_esc50.evaluate(model, mode="linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- ESC50 LINEAR -----------------")
for metric, value in res_esc50.items():
    print (f"{metric}: {value}")

# create a tsv line: model_tag, size, is_linear, dataset_name, accuracy, f1
tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\tesc50\t{res_esc50['accuracy']}\t{res_esc50['f1_score']}")


res_esc50 = evaluator_esc50.evaluate(model, mode="non-linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- ESC50 NON-LINEAR -----------------")
for metric, value in res_esc50.items():
    print (f"{metric}: {value}")
tsv_lines.append(f"{args.model}\t{model_parameters}\tFalse\tesc50\t{res_esc50['accuracy']}\t{res_esc50['f1_score']}")


'''
************************************************************************************************
*                                          FMA-small                                           *
************************************************************************************************
'''

model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=30*16_000)

FMA_DATASET_PATH = "/data1/mlaquatra/datasets/audio_datasets/fma_small/"
FMA_METADATA_PATH = "/data1/mlaquatra/datasets/audio_datasets/fma_metadata/"
evaluator_fma = FMASmall(
    config_path = FMA_METADATA_PATH,
    audio_files_path = FMA_DATASET_PATH,
    verbose=args.verbose
)
res_fma = evaluator_fma.evaluate(model, mode="linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- FMA-small LINEAR -----------------")
for metric, value in res_fma.items():
    print (f"{metric}: {value}")
tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\tfma_small\t{res_fma['accuracy']}\t{res_fma['f1_score']}")

res_fma = evaluator_fma.evaluate(model, mode="non-linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- FMA-small NON-LINEAR -----------------")
for metric, value in res_fma.items():
    print (f"{metric}: {value}")
tsv_lines.append(f"{args.model}\t{model_parameters}\tFalse\tfma_small\t{res_fma['accuracy']}\t{res_fma['f1_score']}")

'''
************************************************************************************************
*                                          RAVDESS                                             *
************************************************************************************************
'''

model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=5*16_000)

RAVDESS_DATASET_PATH = "/data1/mlaquatra/datasets/audio_datasets/ravdess/"
evaluator_ravdess = RAVDESS(path=RAVDESS_DATASET_PATH, verbose=args.verbose)
res_ravdess = evaluator_ravdess.evaluate(model, mode="linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- RAVDESS LINEAR -----------------")
for metric, value in res_ravdess.items():
    print (f"{metric}: {value}")
tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\travdess\t{res_ravdess['accuracy']}\t{res_ravdess['f1_score']}")

res_ravdess = evaluator_ravdess.evaluate(model, mode="non-linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)
print ("----------------- RAVDESS NON-LINEAR -----------------")
for metric, value in res_ravdess.items():
    print (f"{metric}: {value}")
tsv_lines.append(f"{args.model}\t{model_parameters}\tFalse\travdess\t{res_ravdess['accuracy']}\t{res_ravdess['f1_score']}")


# append tsv lines in file
with open(args.tsv_logging_file, "a") as f:
    for line in tsv_lines:
        f.write(line + "\n")

'''
python evaluate_hf_models.py \
    --model facebook/wav2vec2-base \
    --device cuda 
'''