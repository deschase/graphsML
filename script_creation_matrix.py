import numpy as np

def return_tome(namefile):
    data = []
    characters = []
    with open(namefile, "rb") as f:
        list_lines = f.readlines()

        for line in list_lines:
            line = line.decode("utf_8")
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.split(',')
            if len(line) > 0:
                if line[0] != "":
                    data.append(line)
                    for character in line:
                        if not character in characters:
                            characters.append(character)
    f.close()
    return data, characters

def add_new_character(character_name, nb_corres, corres, mat):
    corres[character_name] = nb_corres
    for l in mat:
        l.append(0)
    mat.append([0 for i in range(len(mat[0]))])

def add_characters(characters, corres, mat):
    for character in characters:
        if character not in corres.keys():
            if len(mat) == 0:
                corres[character] = 0
                mat.append([0])
            else:
                cor = len(corres.keys())
                add_new_character(character, cor, corres, mat)


def add_relations(data, corres, mat):
    for together in data:
        for charac1 in together:
            for charac2 in together:
                mat[corres[charac1]][corres[charac2]] += 1

def create_matrix_file(corres, mat, name):
    matr = np.asarray(mat)
    with open("data/" + name + "_matrix.csv", 'w') as f:
        for m in range(matr.shape[0]):
            first = True
            for nb in matr[m,:]:
                if first:
                    f.write(str(nb))
                    first = False
                else:
                    f.write("," + str(nb))
            f.write('\n')

    with open("data/" + name + "_correspondances.csv", 'w') as f:
        for m in corres.keys():
            f.write(m + "," + str(corres[m]) + "\n")





correspondances = dict()
matrix_relation = list()

print "#### You are going to create a matrix of similarities between one piece characters #####"
yes = raw_input("Do you wish to continue? (yes/no)")
if yes == "yes":
    print "Ok perfect !"
    tome_number = raw_input("Enter the volume numbers you want to use (with ',' between each of them) :")
    tome_number = tome_number.replace(' ', '')
    name = tome_number.replace(',', '_')
    tome_number = tome_number.split(',')

    for tome in tome_number:
        print "Loading volume number ",tome
        # we get the list of char on each page and the list of characters in the volume
        data, characters = return_tome("data/tomes/tome" + tome + ".csv")
        # we add the characters not already in the dataset
        add_characters(characters, correspondances, matrix_relation)
        # we count the copresences
        add_relations(data, correspondances, matrix_relation)
    create_matrix_file(correspondances, matrix_relation, name)
else:
    print "BYYYYE !"