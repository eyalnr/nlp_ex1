import os
import math
import random
from collections import Counter, defaultdict

def seq_count(corpus_lines, n):
    """
    Counts how many times each sequence of size n appears in text
    """
    seq_counter = Counter()
    for line in corpus_lines:
        for i in range(0, len(line) - n):
            t = line[i : i + n]
            seq_counter[t] += 1
    return seq_counter

def n_gram_count(corpus_lines, n):
    """
    For each sequence of size n-1, counts characters occurences after the sequence in the text.
    """
    n_counts = defaultdict(Counter)
    for line in corpus_lines:
        for i in range(0, len(line) - n + 1):
            t = line[i : i + n - 1]
            n_char = line[i + n - 1]
            n_counts[t][n_char] += 1
    return n_counts
  
def lm(corpus_file, model_file):
    """
    corpus_file -  a plain text file. Each line should be considered as an individual text (e.g.,
    putting <start> and <end> around it).
    model_file - output file
    
    Creates a lanuage model based on the corpus file and write it to the model file
    """
    with open(corpus_file) as corpus_file_f:
        corpus_data = corpus_file_f.read()
    
    corpus_lines = corpus_data.split('\n')
    
    with open(model_file, 'w') as model_file_f:
        for i in [3, 2, 1]:
            seq_counts = seq_count(corpus_lines, i - 1)
            n_counts = n_gram_count(corpus_lines, i)
       
            for seq in n_counts:
                for c in n_counts[seq]:
                    prob = float(n_counts[seq][c]) / seq_counts[seq]
                    model_file_f.write(seq + c + '\t' + str(prob) + '\n')
                    
            model_file_f.write('\n')

def parse_model_file(model_file_path):
    """
    model_file_path - path to the model file
    
    parse model file and returns a dictionary which maps between sequence to probability
    
    model file format is lines od:
    <char1><char2><char3> <probability>
    ...
    <newline>
    <char1><char2> <probability>
    ...
    <newline>
    <char1> <probability>
    """
    with open(model_file_path) as f:
        model_data = f.read()
    
    lang_model = {}
    
    for line in model_data.split('\n'):
        if line == '':
            continue
        
        seq, prob = line.split('\t', 1)
        
        lang_model[seq] = float(prob)
    return lang_model
    
def eval(input_file, model_file, weights):
    """
    input_file - Text file for evaluation.
    model_file - A model file, as created by lm.
    weights - A list of 3 weights for the 3 models.
    """
    lang_model_probs = parse_model_file(model_file)
    
    with open(input_file) as input_file_f:
        text = input_file_f.read()
    
    probs_interpolation = []
    
    for i in range(2, len(text) + 1):        
        cur_prob_interpolation = 0

        for n in [1, 2, 3]:
            seq = text[i - n + 1 : i + 1]
            
            if not seq in lang_model_probs:
                continue
            
            cur_prob_interpolation += lang_model_probs[seq] * weights[n-1]
            
        probs_interpolation.append(cur_prob_interpolation)
    
    probs = filter(lambda p: p != 0, probs_interpolation)
    
    log_probs = [math.log(prob, 2) for prob in probs]
    # Calculate the perplexity
    perplexity = 2 ** (-sum(log_probs) / len(log_probs))
    
    return perplexity

LANGUAGES = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
START_CHAR = '\x01'
END_CHAR = '\x02'

CORPUS_LINE_PREFIX = START_CHAR * 2
CORPUS_LINE_SUFFIX = END_CHAR * 2 + "\n"

CSV_FILE_SUFFIX = '.csv'
TRAIN_FILE_SUFFIX = '_train'
TEST_FILE_SUFFIX = '_test'
LANG_MODEL_FILE_SUFFIX = '_model'

def parse_tweets_files(dir_path):
    '''
    Parse languages tweets files and build a train and a test sets for each language 
    Each file is a csv where each line format is: tweet_id, tweet_text
    '''
    for lang in LANGUAGES:
        with open(os.path.join(dir_path, lang + CSV_FILE_SUFFIX)) as f:
            data = f.read()
        
        data_lines = data.split('\n')
        
        tweets = []
        for line in data_lines:
            # Adds escaping to tabs in tweets
            line.replace('\t', '\\t')
            
            # line format is: tweet_id, tweet_text
            line_splits = line.split(',', 1)
            
            # handles case where tweets contains newlines
            
            if len(line_splits) != 2:            
                # remove CORPUS_LINE_SUFFIX
                tweets[-1] = tweets[-1][:-len(CORPUS_LINE_SUFFIX)]
                
                # replace newline with space, append line and suffix
                tweets[-1] += " " + line + CORPUS_LINE_SUFFIX
            else:
                # appends 2 start char at start, 2 end chars in the end
                tweets.append(CORPUS_LINE_PREFIX + line_splits[1] + CORPUS_LINE_SUFFIX)
        
        random.shuffle(tweets)
        
        split_index = int(len(tweets) * 0.9)
        
        train_set = tweets[:split_index]
        test_set = tweets[split_index:]
        
        with open(os.path.join(dir_path, lang + TRAIN_FILE_SUFFIX), 'w') as f:
            f.writelines(train_set)
            
        with open(os.path.join(dir_path, lang + TEST_FILE_SUFFIX), 'w') as f:
            f.writelines(test_set)
    
def language_evaluation(dir_path):
    parse_tweets_files(dir_path)
    
    for lang in LANGUAGES:
        lang_train_path = os.path.join(dir_path, lang + TRAIN_FILE_SUFFIX)
        lang_model_path = os.path.join(dir_path, lang + LANG_MODEL_FILE_SUFFIX)
        lm(lang_train_path, lang_model_path)

    weights = [0.3, 0.3, 0.4]
    
    row_header = ' ' * 6 + '\t'
    for lang in LANGUAGES:
        row_header += lang + ' ' * 4 + '\t'
    print row_header
    
    for lang_model in LANGUAGES:
        row = lang_model + '\t'
        lang_model_path = os.path.join(dir_path, lang_model + LANG_MODEL_FILE_SUFFIX)
        for lang_test in LANGUAGES:
            lang_test_path = os.path.join(dir_path, lang_test + TEST_FILE_SUFFIX)
            perplexity = eval(lang_test_path, lang_model_path, weights)
            
            row += "{0:.2f}".format(perplexity) + '\t'
        print row
        
def main():
    language_evaluation(r"data")
    
if __name__ == '__main__':
    main()

    
