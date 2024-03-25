# -*- coding: utf-8 -*-
"""novosadska_baza_podataka.py

Jelenina skripta
lazic.jelenaa@gmail.com

U ovoj skripti vrsena je transkripcija koristenjem google speech sistema.
Koristeno je prvih 300$ koji se dobiju kada se prvi put ulogujes na drive.
Na kraju se dobijaju transkripri snimaka u formi .txt fajlova sacuvani u
transcript_corrected folderu.

"""

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mindful-server-408912-9b007c83d67b.json"

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

# Create a GCS client
storage_client = storage.Client()

# Set your GCS bucket name
bucket_name = "audio_files_srprki_jezik"

# List all objects in the specified bucket
blobs = storage_client.list_blobs(bucket_name)

# Dictionary to store file paths for each user and folder
wav_files_by_user_folder = {}

# Iterate through all objects in the bucket
for blob in blobs:
    if blob.name.lower().endswith('.wav'):
        # Extract user and folder names from the object's path
        user_folder_name, folder_name = blob.name.split('/')[0:2]

        # Initialize an empty dictionary if the user_folder is encountered for the first time
        if user_folder_name not in wav_files_by_user_folder:
            wav_files_by_user_folder[user_folder_name] = {}

        # Initialize an empty list if the folder is encountered for the first time
        if folder_name not in wav_files_by_user_folder[user_folder_name]:
            wav_files_by_user_folder[user_folder_name][folder_name] = []

        # Append the file path to the list
        wav_files_by_user_folder[user_folder_name][folder_name].append(blob.name)

# Create a SpeechClient
client = speech.SpeechClient()
language_code = 'sr-Latn'

# Configure the recognition request
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code=language_code,
    enable_word_time_offsets=True,
)

def process_transcript(audio_path, response):

    text_path = audio_path[:-4]
    output_file_name = os.path.join('..','podaci', 'transcript_corrected', f"{text_path}_transcript.txt")


    # Print the recognized words and timestamps
    for result in response.results:
        alternative = result.alternatives[0]
        with open(output_file_name, 'w') as output_file:
            output_file.write(f"Transcript: {result.alternatives[0].transcript}\n")

        for word_info in alternative.words:
            start_time = (
                word_info.start_time.seconds
                + word_info.start_time.microseconds * 1e-6
            )
            end_time = (
                word_info.end_time.seconds
                + word_info.end_time.microseconds * 1e-6
            )
            word = word_info.word
            # Redirecting the output to a text file
            with open(output_file_name, 'a') as output_file:
                output_file.write(f"Word: {word}, start: {start_time}, end: {end_time}\n")

        # Printing a message indicating where the output is saved
        #print(f"Output saved to {output_file_name}")

# Print or process the collected files by folder
for user_folder, folders in wav_files_by_user_folder.items():
    print(f"User: {user_folder}")
    print(f"Number of folders: {len(folders)}")

    # Create folder on drive
    folder_path = os.path.join('..','podaci', 'transcript_corrected', f"{user_folder}")
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
      print(f"Folder created: {folder_path}")


    for folder, files in folders.items():
      print(f"Folder: {folder}")
      print(f"Number of files: {len(files)}")

      # Create folder on drive
      folder_path = os.path.join('..','podaci', 'transcript_corrected', f"{user_folder}", f"{folder}")
      # Create the folder if it doesn't exist
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")

      for audio_path in files:
        # Configure the audio input
        audio = speech.RecognitionAudio(uri=f"gs://{bucket_name}/{audio_path}")
        # Perform the speech recognition
        try:
          response = client.recognize(config=config, audio=audio)
          process_transcript(audio_path, response)
        except:
          print(audio_path)