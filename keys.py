def getAlphabet(type = 'English'):
    if type == 'English':
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    else:
        alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯаоуыэяёюиебпвфдтзсгкхмнлрцжшщчйъь- '
    return alphabet