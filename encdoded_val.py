def converted_value():

	return converted_to_number
conv = []
with open('listfile.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        conv.append(currentPlace)
def conv_test():
	return conv