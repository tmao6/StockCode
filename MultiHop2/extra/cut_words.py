


file_object  = open('cut_words.txt', 'r')
fh = open('../core/finance_words.txt', 'w')

Lines = file_object.readlines()

# Strips the newline character
for line in Lines:
    if line != '\n':
        sent = line.strip()
        substring1 = sent.split("-", 1)[0] #removes all the dashes
        substring1 = substring1.split("–", 1)[0] #removes weird character
        substring1 = substring1.split("(", 1)[0] #removes paranthesis
        substring1 = substring1.split("/", 1)[0] #removes forward slash
        substring1 = substring1.split("®", 1)[0] #removes forward slash





        if len(substring1) > 5:
            print(substring1)
            fh.write(substring1)
            fh.write('\n')

file_object.close()
fh.close()


