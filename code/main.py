from DataUtils import DataFilter
import json
from os import listdir
from os.path import isfile, join

# massager = DataFilter(
#     '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/Video_Games.json',
#     '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/Video_Games_filtered.json')
# massager.process_file()





# string = '{"asin": "B0000ALPBP", "questions": [{"questionType": "open-ended", "askerID": "A2SW1WZZ6RU4UV", "questionTime": "November 20, 2013", "questionText": "Will it work with newest macbook pro retina? It only has a headphone out port. Will I be able to use the mic as well?", "answers": [{"answerText": "macbook uses a 3.5mm jack, and this plantronics is a 2.5mm jack so its not compatible, macbook internal mic is very sensitive so you dont need an external mic but if you want a mic you can see the macbook compatible accessories", "answererID": "A2MYRU5GAAQ64N", "answerTime": "November 20, 2013", "helpful": [1, 1]}, {"answerText": "I dont know if it is compatible with macbook pro retina. I wouldnt recommend it, it seems cheaply built.", "answererID": "A3CUABBX07IJB1", "answerTime": "November 20, 2013", "helpful": [0, 1]}, {"answerText": "The only device I tried it with was a 2 line analog phone. I could hear fine with it but my outgoing (mic) volume was very low. I ended up buying a new headset elsewhere", "answererID": "AN4CX1N21OMKL", "answerTime": "November 20, 2013", "helpful": [0, 1]}]}, {"questionType": "open-ended", "askerID": "A3KKV969TP8ER8", "questionTime": "May 14, 2014", "questionText": "Will this item also allow for you to listen to music while not on a call.", "answers": [{"answerText": "Again, this headset is designed to work with a cordless instrument connected to a landline. Unless you have music piped into your landline instrument... then NO. This product is NOT intended for use with a cellphone.", "answererID": "ATVOWL1L1JHGU", "answerTime": "May 14, 2014", "helpful": [1, 1]}, {"answerText": "I do not know. The ones I bought were awful compared to the original.", "answererID": "A3276XNFRDGHJ3", "answerTime": "May 14, 2014", "helpful": [0, 0]}, {"answerText": "When I purchase this item. It allows you to listen to music and hear callers. It does not allow your calls to hear you. It is not worth it.", "answererID": "A2Q5AYJ8D0U3K1", "answerTime": "May 14, 2014", "helpful": [0, 0]}]}, {"questionType": "yes/no", "askerID": "AX4L6DVN3E0XX", "questionTime": "November 15, 2014", "questionText": "in the description it said,  Samsung ready,   well Im not tech Savoy,   but it didnt fit my s5g. Samsung galaxy 5. is there an adapter ?", "answers": [{"answerText": "No", "answererID": "A1424MYNPILK3K", "answerTime": "November 15, 2014", "helpful": [0, 0], "answerType": "N", "answerScore": "0.0366"}, {"answerText": "U ordered the wrong size... you need to get the 3.5.. I have used the 3.5 in all my phones for well over 10 years... no other type beats these out...", "answererID": "A1LVGFOTVXC3CY", "answerTime": "November 25, 2014", "helpful": [0, 0], "answerType": "?", "answerScore": "0.5722"}]}, {"questionType": "open-ended", "askerID": "A1M70QDA3YMJWO", "questionTime": "March 30, 2014", "questionText": "Can u sing/perform on stage with these?", "answers": [{"answerText": "Probably not. In fact these are poor quality compared to the last set i bought.", "answererID": "A3276XNFRDGHJ3", "answerTime": "March 30, 2014", "helpful": [0, 0]}, {"answerText": "Ha! I never tried it.... let me know if you do.. ok? I have only used this Bluetooth headset to talk and receive calls on my home cordless phone. If you clipped the wire onto your lapel to secure it I bet you could dance & sing along w/o a problem. if you try it and it works let me know.. have fun!", "answererID": "A3R9FIX14THM8P", "answerTime": "March 30, 2014", "helpful": [0, 0]}, {"answerText": "Seriously? Are you seriously thinking through this, Sonya? This $5.00 headset is designed to be used on a cordless phone. Even if you could anchor the phone handset to your person, and secure the headset so it would stay in place throughout the activity and movement associated with performance, the audio quality is simply not there. This unit was not intended for such use. I read your question and thought to myself, it must be April 1st.", "answererID": "ATVOWL1L1JHGU", "answerTime": "March 31, 2014", "helpful": [0, 0]}]}, {"questionType": "yes/no", "askerID": "A1NCJ6NOOXCN45", "questionTime": "December 26, 2013", "questionText": "will this work for ps3?", "answers": [{"answerText": "This is for a phone.I wouldnt use this on a ps3.", "answererID": "AJM91SSS591XE", "answerTime": "December 26, 2013", "helpful": [0, 0], "answerType": "?", "answerScore": "0.7025"}, {"answerText": "Sorry, no, it will not", "answererID": "A24QRVL578V1JO", "answerTime": "December 27, 2013", "helpful": [0, 0], "answerType": "N", "answerScore": "0.0638"}, {"answerText": "I cannot say if it would work for ps3.  It did not work at all for us, and we purchased it to use with a telephone (as it is intended).  A huge disappointment,and I would not recommend.", "answererID": "A2H27KRDFBJ827", "answerTime": "December 27, 2013", "helpful": [0, 0], "answerType": "?", "answerScore": "0.8201"}]}]}';
#
# parsed_string = json.loads(string)
#
# questions = parsed_string['questions']
#
# print len(questions)
#
# for question in questions:
#     print question['questionText']





#

f_out_q = open(
    '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/questions.txt',
    'w')
f_out_a = open(
    '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna/answers.txt',
    'w')


def extract_qna(input_file_name):
    f_in = open(input_file_name, 'r')

    print 'Processing...' + input_file_name
    _line = f_in.readline()
    while _line:
        # print _line
        # print _line.isspace()
        # print '-----'
        if _line.isspace():
            _line = f_in.readline()
            continue

        try:
            parsed_line = json.loads(_line)
        except Exception, e:
            # print 'Exception encountered!'
            _line = f_in.readline()
            continue
        questions = parsed_line['questions']
        for question in questions:
            try:
                f_out_q.write(question['questionText'] + '\n')
            except Exception, e:
                pass
            answers = question['answers']
            for answer in answers:
                try:
                    f_out_a.write(answer['answerText'] + '\n')
                except Exception, e:
                    pass
        _line = f_in.readline()
    f_in.close()
    # f_out_a.close()
    # f_out_q.close()
    print 'QnA_extracted.'


mypath = '/Users/sampanna.kahu/Desktop/OldMacBackup/Desktop/PersonalWorkspace/ml/cnn-text-classification-tf/data/qna'
onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and '_filtered' in f)]
for file1 in onlyfiles:
    input_file = join(mypath, file1)
    print input_file
    extract_qna(input_file)
