def remove_spec_chars(line1):
    line1 = line1.replace("\"", "'")
    line1 = line1.replace("{'", "{\"")
    line1 = line1.replace("':", "\":")
    line1 = line1.replace(" u'", " \"")
    line1 = line1.replace("',", "\",")
    line1 = line1.replace(", '", ", \"")
    line1 = line1.replace(": '", ": \"")
    line1 = line1.replace("u\"", "\"")
    line1 = line1.replace("'}", "\"}")
    line1 = line1.replace("'", "")
    line1 = line1.replace("\\\\[a-z]+", "")
    return line1
