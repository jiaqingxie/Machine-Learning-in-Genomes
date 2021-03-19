def BF(text, pattern):
    i = 0  #text_start_sign
    j = 0  #pattern_start_sign
    while i < len(text) and j < len(pattern):
        if(text[i] == pattern[j]):
            i = i + 1
            j = j + 1
        else:
            i = i - j + 1
            j = 0
    if (j == len(pattern)):
        return 1
    else:
        return 0 
