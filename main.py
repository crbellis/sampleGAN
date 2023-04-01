import librosa
import soundfile as sf
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

N_MELS = 128

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# audio 
audio, sr = librosa.load("./data/909_HatClosed04.wav")

# Preprocess the input text using the tokenizer
input_text = "909 hi hat"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)

inputs = tf.keras.Input(shape=(6, ), dtype="int32")
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
x = bert_model(inputs)
x = x[1]
x = tf.keras.layers.Dense(2048, activation="relu")(x)
audio_output_layer = tf.keras.layers.Dense(N_MELS*10, activation='linear')(x)
model = tf.keras.Model(inputs=inputs, outputs=audio_output_layer)

model.compile(loss='mean_squared_error', optimizer='adam')

mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)

audio_features = np.array([mel_spec])
audio_features = audio_features.reshape(-1, N_MELS*10)
input_data = np.array([input_ids])
model.fit(input_data, audio_features, batch_size=1, epochs=30)

predicted_audio = model.predict(np.array([input_ids]))
predicted_audio = predicted_audio.reshape(-1, N_MELS, 10)
predicted_audio_waveform = librosa.feature.inverse.mel_to_audio(predicted_audio)
sf.write(f"predicted_audio_{N_MELS}.wav", predicted_audio_waveform[0], sr, subtype="PCM_24")
