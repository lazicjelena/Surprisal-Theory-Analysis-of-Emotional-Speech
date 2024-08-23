# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:18:14 2024

@author: Jelena
"""
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import os

duration_list = []
user_list = []
gender_list = []
emotion_list = []
target_sentence_list = []

# Define the directory path
folder_directory_path = os.path.join('..','podaci', 'data_mono')
transcript_directory_path = os.path.join('..','podaci', 'transcript_corrected')

gender_directory_path = os.path.join('..','podaci', 'gender_data.csv')
gender_df = pd.read_csv(gender_directory_path)

# Loop through the directory
for user in os.listdir(folder_directory_path):
    print(user)
    
    for emotion in [0,1,2,3,4]:
        directory_path = os.path.join(folder_directory_path, user, str(emotion))  
        
        if os.path.exists(directory_path):
            print(emotion)
            
            for filename in os.listdir(directory_path):
                if filename.endswith('.wav'):
                    
                    try:
                        file_path = os.path.join(directory_path, filename)
                        # Load the audio file
                        y, sr = librosa.load(file_path, sr=None)
                        transcript_path = os.path.join(transcript_directory_path, user, str(emotion), filename[:-4] + '_transcript.txt')
                        with open(transcript_path, 'r', encoding='utf-8') as file:
                            first_line = file.readline().strip()
                            
                        target_sentence_list.append(first_line[12:])
                        emotion_list.append(emotion)
                        user_list.append(emotion)
                        gender_list.append(gender_df[gender_df['Speaker'] == int(user)]['Gender'].values[0])
                        duration_list.append(len(y))
                    except:
                        print(f"Error occured while procesing file {filename}")
                        

time_list = []
for duration in duration_list:
    time = duration / 44100
    time_list.append(time)

word_num_list = []
str_length_list = []
for sentence in target_sentence_list:
    str_length_list.append(len(sentence))
    words = sentence.split(' ')
    word_num_list.append(len(words))
    
avarge_speech_rate_list = []
for time, length in zip(time_list,str_length_list):
    av_speach_time = time/length
    avarge_speech_rate_list.append(av_speach_time)
    
# Create DataFrame
df = pd.DataFrame({'gender':gender_list ,
                   'emotion': emotion_list, 
                   'time': time_list, 
                   'word num': word_num_list, 
                   'length': str_length_list,
                   'av speech rate': avarge_speech_rate_list})





plt.figure(figsize=(14, 8))  # Adjust figure size for better spacing
ind = 1

# Define the labels for the emotions
emotion_labels = ['neutral', 'happy', 'sad', 'scared', 'angry']

for gender in ['f', 'm']:
    data = df[df['gender'] == gender]
    plt.subplot(1, 2, ind)
    ind += 1
    print(f"Gender {gender}")
    
    for emotion in [0, 1, 2, 3, 4]:
        pom_emotion = data[data['emotion'] == emotion]
        x_axis = []
        y_axis = []
        #print(f"Emotion {emotion}")

        for x in range(1, 15):
            pom = pom_emotion[pom_emotion['word num'] == x]
            if len(pom) > 0:
                x_axis.append(x)
                rates = pom['av speech rate'].values
                rates = rates[rates <= 0.5]
                y_axis.append(np.mean(rates))
                
        plt.plot(x_axis, y_axis, linewidth= 3, marker='o', markersize = 10, label=emotion_labels[emotion])
        print(f"{np.nanmean(y_axis):.3f} ({np.nanstd(y_axis):.3f})")
        
    plt.ylim([0.05, 0.16])
    # Add legend, place it outside the plots
    plt.legend(fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Add subplot titles
    title = 'Female' if gender == 'f' else 'Male'
    plt.title(title, fontsize=25)

# Add common x-axis label
plt.text(0.5, 0.035, 'Number of words per sentence', ha='center', fontsize=25)
#plt.text(0.0001, 0.5, 'Speech rate', va='center', rotation='vertical', fontsize=25)

# Add main title for the whole figure
plt.suptitle('Speech Rate by Gender and Emotion', fontsize=30)

# Ensure there's no overlap
plt.tight_layout(rect=[0.05, 0.05, 0.1, 0.95])

plt.show()




plt.figure(figsize=(14, 8))  # Adjust figure size for better spacing
ind = 1

# Define the labels for the emotions
emotion_labels = ['неутрално', 'срећно', 'тужно', 'уплашено', 'љуто']

for gender in ['f', 'm']:
    data = df[df['gender'] == gender]
    plt.subplot(1, 2, ind)
    ind += 1
    print(f"Gender {gender}")
    
    for emotion in [0, 1, 2, 3, 4]:
        pom_emotion = data[data['emotion'] == emotion]
        x_axis = []
        y_axis = []
        #print(f"Emotion {emotion}")

        for x in range(1, 15):
            pom = pom_emotion[pom_emotion['word num'] == x]
            if len(pom) > 0:
                x_axis.append(x)
                rates = pom['av speech rate'].values
                rates = rates[rates <= 0.5]
                y_axis.append(np.mean(rates))
                
        plt.plot(x_axis, y_axis, linewidth= 3, marker='o', markersize = 10, label=emotion_labels[emotion])
        print(f"{np.nanmean(y_axis):.3f} ({np.nanstd(y_axis):.3f})")
        
    plt.ylim([0.05, 0.16])
    # Add legend, place it outside the plots
    plt.legend(fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Add subplot titles
    title = 'Жене' if gender == 'f' else 'Мушкарци'
    plt.title(title, fontsize=25)

# Add common x-axis label
plt.text(0.5, 0.035, 'Број ријечи у реченици', ha='center', fontsize=25)
#plt.text(0.0001, 0.5, 'Speech rate', va='center', rotation='vertical', fontsize=25)

# Add main title for the whole figure
plt.suptitle('Брзина изговора', fontsize=30)

# Ensure there's no overlap
plt.tight_layout(rect=[0.05, 0.05, 0.1, 0.95])

plt.show()






























