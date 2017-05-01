import pandas as pd

questions = pd.read_csv(
    '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/questions.txt',
    error_bad_lines=False, header=None, sep='\n')

print questions.describe
print questions.shape

answers = pd.read_csv(
    '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/answers.txt',
    error_bad_lines=False, header=None, sep='\n')

print answers.describe
print answers.shape
