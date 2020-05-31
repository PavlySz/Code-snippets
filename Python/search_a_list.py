def search_list(substring, my_list):
    '''
    Search a list by a substring

    Args:
        - substring (string):  value to search the list by
        - my_list (list): list to search

    Returns:
        - values that match the substring in a list
    '''
    suggestions = [title for title in my_list if title.lower().find(substring.lower()) != -1]

    return suggestions

