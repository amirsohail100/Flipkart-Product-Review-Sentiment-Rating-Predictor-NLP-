def remove_emoj(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new+=i
    return new

def remove_num(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new+=i
    return new


import string

def remove_punc(txt):
  return txt.translate(str.maketrans('','',string.punctuation))


# def remove(txt):
#     words = txt.split()
#     cleaned = []
#     for i in words:
#         if not i in s_W:
#             cleaned.append(i)
#     return ' '.join(cleaned)


def remove_other(txt):
    new = ""
    l = ["!","@","#","$","%","^","&","*","?"]
    for i in txt:
        if i not in l:
            new+=i
    return new


