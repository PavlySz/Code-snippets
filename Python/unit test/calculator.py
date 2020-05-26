def add(x, y):
    '''Addition function'''
    return x+y 


def sub(x, y):
    '''Subtraction function'''
    if (type(x) not in [int, float]) or (type(y) not in [int, float]):
        raise TypeError("Can't substract strings, mate!")
    return x-y 


def mul(x, y):
    '''Multiplication function'''
    if (type(x) not in [int, float]) and (type(y) not in [int, float]):
        raise TypeError("Can't multiply two strings, mate!")
    return x*y 


def div(x, y):
    '''Division function'''
    if (type(x) not in [int, float]) or (type(y) not in [int, float]):
        raise TypeError("Can't divide strings, mate!")
    elif x != 0 and y == 0:
        raise ZeroDivisionError("Can't divide by Zero, ya dingus!")
    elif x == 0 and y == 0:
        return 0
    return x/y 


