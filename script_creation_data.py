import os

def returnNbLines(namefile):
    nblines = 0
    with open(namefile, "rb") as f:
        list_lines = f.readlines()

        for line in list_lines:
            line = line.decode("utf_8")
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.split(',')
            if len(line) > 0:
                if line[0] != "":
                    nblines += 1.
    f.close()
    return nblines

def writeNames(file, names):
    file.write(names + '\n')

tome_number = 0
print "#### You are going to enter new data into the one piece character description #####"
yes = raw_input("Do you wish to continue? (yes/no)")
if yes == "yes":
    print "Ok perfect !"
    needToEnter = True
    while needToEnter:
        try:
            tome_number = int(raw_input("Enter the number of the volume you have in your hands :"))
            if tome_number > 0 and tome_number < 26:
                needToEnter = False
            else:
                print "Out of the desired ones"
        except ValueError:
            print "Not a volume"

    line = 0
    if not os.path.isfile("data/tomes/tome" + str(tome_number) + ".csv"):
        print "##### You are starting this volume #####"
        f = open("data/tomes/tome" + str(tome_number) + ".csv", 'w')
    else:
        line = returnNbLines("data/tomes/tome" + str(tome_number) + ".csv")
        print "The number of pages registered for this volume is already :", int(line)
        f = open("data/tomes/tome" + str(tome_number) + ".csv", 'a')

    finished = False
    while not finished:
        line += 1
        print "Enter the names of the characters in page ", int(line), " with ',' between each name (end if finished)"
        names = raw_input("names: ")
        if names != "end":
            writeNames(f, names)
        else:
            qut = raw_input("Do you wish to quit this volume and the script ? (yes/no)")
            if qut == "yes":
                print "OK ! BYYYYE !"
                f.close()
                finished = True
            else:
                "Ok let's continue"
                line -= 1

else:
    print "BYYYYYYYE !"
