import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

class STT_model(object):

    def __init__(self):
        #load model and tokenizer
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h")
        print("Model loaded successfully")

    def convert(self, filepath):
        #load any audio file of your choice
        speech, rate = librosa.load(filepath, sr=16000)

        input_values = self.tokenizer(speech, return_tensors = 'pt').input_values
        #Store logits (non-normalized predictions)
        logits = self.model(input_values).logits

        #Store predicted id's
        predicted_ids = torch.argmax(logits, dim =-1)
        #decode the audio to generate text
        transcriptions = self.tokenizer.decode(predicted_ids[0])
        return transcriptions

if __name__ == '__main__':
    model = STT_model()
    transcription = model.convert('male.wav')
    print(transcription)
