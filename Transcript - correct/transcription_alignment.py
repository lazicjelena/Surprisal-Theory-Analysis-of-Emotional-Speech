# -*- coding: utf-8 -*-
"""transcription_alignment.ipynb

Jelenina skripta
lazic.jelenaa@gmail.com

Skripta vrsi korekciju transkripta na osnovu target recenica.
"""
from difflib import SequenceMatcher
import pandas as pd
from fuzzywuzzy import fuzz
import os

target_sentences_file_path = os.path.join('..','podaci','target_sentences.csv')
target_sentences_df = pd.read_csv(target_sentences_file_path)

wrong_transcription_file_path = os.path.join('..','podaci', 'wrong_transcription.csv')
wrong_transcription_df = pd.read_csv(wrong_transcription_file_path)

def align_words(sentence1, sentence2):

    # Tokenize sentences into lowercase words
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Use SequenceMatcher to get matching blocks
    matcher = SequenceMatcher(None, words1, words2)
    matching_blocks = matcher.get_matching_blocks()

    # Extract aligned words
    aligned_words = []

    for block in matching_blocks:
        start_idx1, start_idx2, block_size = block
        aligned_words.extend(zip(words1[start_idx1:start_idx1 + block_size], words2[start_idx2:start_idx2 + block_size]))

    return aligned_words

def sum_position_of_the_same_chars(word1, word2):
    # Ensure both words are of the same length
    min_len = min(len(word1), len(word2))
    word1 = word1[:min_len]
    word2 = word2[:min_len]

    # Calculate the sum of positions of differing characters
    sum_positions = 0
    for c1, c2 in zip(word1[:min_len], word2[:min_len]):
      if c1 == c2:
        sum_positions +=1

    return sum_positions + len(word1) + len(word2) - 2*min_len

def pair_words_with_difference(sentence, target_sentence, max_difference=2):
    # Tokenize sentences into lowercase words
    words1 = sentence.split()
    words2 = target_sentence.split()

    # Group unmatched words that have all except one character the same
    grouped_words = []
    paired_words1 = set()
    paired_words2 = set()

    for word1 in words1:
        for word2 in words2:
            if word1 in paired_words1:
              continue  # Skip already paired word1
            if word2 in paired_words2:
                continue  # Skip already paired word2

            sim_count = sum_position_of_the_same_chars(word1, word2)
            if max_difference >= max(len(word1), len(word2)) - sim_count:
                grouped_words.append((word1, word2))
                paired_words1.add(word1)
                paired_words2.add(word2)

    return grouped_words

def align_endpoints(sentence1, sentence2, aligned_words):

    # Get the indices of aligned words in each sentence
    aligned_indices_s1 = []
    aligned_indices_s2 = []
    max_number1 = -1
    max_number2 = -1
    for pair in aligned_words:
      indices_s1 = [index for index, word in enumerate(sentence1.split()) if word == pair[0]]
      indices_s2 = [index for index, word in enumerate(sentence2.split()) if word == pair[1]]
      # Ensure the number added to aligned_indices_s1 is larger than any number in the list
      smallest_number1 = [num for num in indices_s1 if num > max_number1]
      smallest_number2 = [num for num in indices_s2 if num > max_number2]
     # Add the larger_number to aligned_indices_s1
      aligned_indices_s1.append(min(smallest_number1))
      aligned_indices_s2.append(min(smallest_number2))
     # Find the minimum number from indices_s1
      max_number1 = max(aligned_indices_s1)
      max_number2 = max(aligned_indices_s2)

    # Get the words before the first aligned word in each sentence
    unaligned_words_s1_start = sentence1.split()[:min(aligned_indices_s1)]
    unaligned_words_s2_start = sentence2.split()[:min(aligned_indices_s2)]

    # Get the words after the last aligned word in each sentence
    unaligned_words_s1_end = sentence1.split()[max(aligned_indices_s1) + 1:]
    unaligned_words_s2_end = sentence2.split()[max(aligned_indices_s2) + 1:]

    # Convert tuples to lists for modification
    aligned_words_list = [list(pair) for pair in aligned_words]

    # Modify the first pair if needed
    left_start_s1 = ' '.join(unaligned_words_s1_start)
    left_start_s2 = ' '.join(unaligned_words_s2_start)
    if left_start_s1 != '' and left_start_s2 == '':
        aligned_words_list[0][0] = left_start_s1 + ' ' + aligned_words_list[0][0]
    if left_start_s2 != '' and left_start_s1 == '':
        aligned_words_list[0][1] = left_start_s2 + ' ' + aligned_words_list[0][1]

    # Modify the last pair if needed
    left_end_s1 = ' '.join(unaligned_words_s1_end)
    left_end_s2 = ' '.join(unaligned_words_s2_end)
    if left_end_s1 != '' and left_end_s2 == '':
        aligned_words_list[-1][0] = aligned_words_list[-1][0] + ' ' + left_end_s1
    if left_end_s2 != '' and left_end_s1 == '':
        aligned_words_list[-1][1] = aligned_words_list[-1][1] + ' ' + left_end_s2

    # Convert lists back to tuples
    aligned_words_list = [tuple(map(str.strip, pair)) for pair in aligned_words_list]
    aligned_words = [tuple(pair) for pair in aligned_words_list]

    # Create extended aligned pairs
    extended_aligned_pairs_start = [(' '.join(unaligned_words_s1_start), ' '.join(unaligned_words_s2_start))]
    extended_aligned_pairs_end = [(' '.join(unaligned_words_s1_end), ' '.join(unaligned_words_s2_end))]
    extended_aligned_pairs = extended_aligned_pairs_start + aligned_words + extended_aligned_pairs_end

    # Filter out empty pairs
    extended_aligned_pairs = [pair for pair in extended_aligned_pairs if all(word != '' for word in pair)]

    return extended_aligned_pairs

def align_middlepoints(sentence1, sentence2, extended_aligned_pairs):
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()

    paired_words1 = [word_pair[0] for word_pair in extended_aligned_pairs]
    paired_words2 = [word_pair[1] for word_pair in extended_aligned_pairs]

    words1 = sentence1.split()
    words2 = sentence2.split()

    ind_w1 = 0
    ind_w2 = 0
    ind_pw1 = 0
    ind_pw2 = 0

    final_pair1 = []
    final_pair2 = []
    pom1 = ''
    pom2 = ''
    while ind_w1 < len(words1) or ind_w2 < len(words2) or ind_pw1 < len(paired_words1) or ind_pw2 < len(paired_words2):
      cond1 = words1[ind_w1] == paired_words1[ind_pw1] or (pom1 + ' ' + words1[ind_w1]).strip() == paired_words1[ind_pw1]
      cond2 = words2[ind_w2] == paired_words2[ind_pw2] or (pom2 + ' ' + words2[ind_w2]).strip() == paired_words2[ind_pw2]
      if cond1 and cond2:
        if pom1 == '' or pom2 == '':
          final_pair1.append((pom1 + ' ' + words1[ind_w1]).strip())
          final_pair2.append((pom2 + ' ' + words2[ind_w2]).strip())
        else:
          final_pair1.append(pom1)
          final_pair2.append(pom2)
          final_pair1.append(words1[ind_w1])
          final_pair2.append(words2[ind_w2])
        ind_w1 += 1
        ind_w2 += 1
        ind_pw1 += 1
        ind_pw2 += 1
        pom1 = ''
        pom2 = ''
      else:
        if not cond2:
          pom2 += ' ' + words2[ind_w2]
          pom2 = pom2.strip()
          ind_w2 += 1
        if not cond1:
          pom1 += ' ' + words1[ind_w1]
          pom1 = pom1.strip()
          ind_w1 += 1


    # Create pairs of unmatched words
    final_pairs = list(zip(final_pair1, final_pair2))

    return final_pairs

def align_transcript(sentence, target_sentence):

  sentence = sentence.lower()
  target_sentence = target_sentence.lower()

  # alignment of the same utterance units
  aligned_words = align_words(sentence, target_sentence)
  # correct begining and end ofutterance
  if aligned_words == []:
    print('Failed to match, second try')
    aligned_words = pair_words_with_difference(sentence, target_sentence)
  print(aligned_words)
  extended_aligned_pairs = align_endpoints(sentence, target_sentence, aligned_words)
  print(extended_aligned_pairs)
  # correct middle point words
  final_aligned_pairs = align_middlepoints(sentence, target_sentence, extended_aligned_pairs)

  return final_aligned_pairs

def find_target_sentence(sentence, df):
    max_similarity = 0
    target_sentence = ""

    for _, row in df.iterrows():
        current_similarity = fuzz.ratio(sentence, row['Text'])
        if current_similarity > max_similarity:
            max_similarity = current_similarity
            target_sentence = row['Text']

    return target_sentence

def correct_sentence(sentence):
    target_sentence = find_target_sentence(sentence, target_sentences_df)
    align_pairs = align_transcript(sentence, target_sentence)
    return target_sentence, align_pairs

def create_dictionary_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extracting transcript and word information
    transcript_text = lines[0].strip()[12:]
    words_data = []

    for line in lines[1:]:
        parts = line.strip().split(', ')
        word = parts[0].split(': ')[1]
        start = float(parts[1].split(': ')[1])
        end = float(parts[2].split(': ')[1])

        word_info = {"Word": word, "start": start, "end": end}
        words_data.append(word_info)

    # Creating the dictionary
    transcript_data = {"Transcript": transcript_text, "Words": words_data}

    return transcript_data

def correct_transcript(file_path):

    transcript_data = create_dictionary_from_file(file_path)
    new_transcript_data = {
        "Words": []
    }

    first_line = transcript_data['Transcript'].strip()
    print(first_line)
    is_sentence_in_wrong_transcription = first_line in wrong_transcription_df['Wrong Sentences'].values
    if is_sentence_in_wrong_transcription:
      print('Wrong Transcription!')
      return new_transcript_data

    transcript_data['Transcript'] = transcript_data['Transcript'].lower()
    first_line = transcript_data['Transcript'].strip()
    target_sentence, aligned_pairs = correct_sentence(first_line)
    target_sentence = target_sentence.lower()

    # Loop through the words in the dictionary and update with aligned pairs
    word_pom = ''
    start_time_pom = ''
    ind_word_info = 0
    ind_aligned_pair = 0

    while ind_word_info < len(transcript_data["Words"]) and ind_aligned_pair < len(aligned_pairs):
        word_info = transcript_data["Words"][ind_word_info]
        word_info['Word'] = word_info['Word'].lower()
        aligned_pair = aligned_pairs[ind_aligned_pair]
        if word_info["Word"] == aligned_pair[0] or word_pom + ' ' + word_info["Word"] == aligned_pair[0]:
          new_word_info = {"Word": aligned_pair[1], "start": word_info["start"], "end": word_info["end"]}
          if start_time_pom != '':
            new_word_info["start"] = start_time_pom
            start_time_pom = ''
            word_pom = ''
          # Append the new word dictionary to the "Words" list
          new_transcript_data["Words"].append(new_word_info)
          ind_aligned_pair += 1
        else:
          if word_pom == '':
            start_time_pom =  word_info["start"]
          word_pom = (word_pom + ' ' + word_info["Word"]).strip()
        ind_word_info += 1

    # Update transcript
    transcript_data['Transcript'] = target_sentence
    transcript_data['Words'] = new_transcript_data['Words']

    return transcript_data

def write_transcript(output_file_path, transcript_data):

    with open(output_file_path, 'w', encoding='utf-8') as file:
        # Writing transcript text
        file.write(f"Transcript: {transcript_data['Transcript']}\n")
        # Writing word information
        for word_info in transcript_data['Words']:
            file.write(f"Word: {word_info['Word']}, start: {word_info['start']}, end: {word_info['end']}\n")
    return

def process_folder(input_folder, output_folder):

    err = 0
    processed_folders = set()

    for root, dirs, files in os.walk(input_folder):
      for filename in files:
        try:
          file_path = os.path.join(root, filename)
          print(file_path)
          new_text = correct_transcript(file_path)
          if len(new_text['Words']) == 0:
            print(f"Error processing: {file_path}")
            err += 1
            continue

          # Creating the corresponding directory structure in the output path
          relative_path = os.path.relpath(file_path, input_folder)
          output_path = os.path.join(output_folder, relative_path)
          output_dir = os.path.dirname(output_path)

          # Print a message when a new folder is create
          current_folder = os.path.dirname(relative_path)
          if current_folder not in processed_folders:
            print(f"Created folder: {current_folder}")
            processed_folders.add(current_folder)

          # Create the corresponding directory in the output path if it doesn't exist
          os.makedirs(output_dir, exist_ok=True)

          # Write the corrected transcript to the output path
          write_transcript(output_path, new_text)

        except:
          # Open the file in append mode (creates the file if it doesn't exist)
          with open('/content/drive/MyDrive/PhD/Transcript - correct/transcript_corrected/errors.txt', 'a') as file:
          # Add new text lines
            lines_to_add = [file_path]
            file.write("\n".join(lines_to_add))

    print('Number of errors: ' + str(err))
    return

# Example usage:
input_directory = os.path.join('..','podaci','transcript')
output_directory = os.path.join('..','podaci','transcript_corrected')

process_folder(input_directory, output_directory)