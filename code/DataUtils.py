import fileutil
import json


class DataFilter:
    def __init__(self, input_file_path, output_file_path):
        self.f_in = open(input_file_path, "r")
        self.f_out = open(output_file_path, "w")

    def process_file(self):
        print 'Processing...'
        _line = self.f_in.readline()
        while _line:
            # print _line
            _line_out = fileutil.remove_spec_chars(_line)
            # print _line_out
            self.f_out.write(_line_out + '\n')
            # print '---------------------------------------------------------------'
            _line = self.f_in.readline()
        self.f_in.close()
        self.f_out.close()
        print 'File processed.'
