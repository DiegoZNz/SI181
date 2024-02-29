from random import choice

options = ["piedra", "tijeras", "papel"]

#Generamos una funci[on con todas las combinaciones posibles
def search_winner(player1, player2):
    if player1 == player2:
        result = 0
    elif player1 == "piedra" and player2 == "tijeras":
        result = 1
    elif player1 == "piedra" and player2 == "papel":
        result = 2
    elif player1 == "tijeras" and player2 == "piedra":
        result = 2
    elif player1 == "tijeras" and player2 == "papel":
        result = 1
    elif player1 == "papel" and player2 == "piedra":
        result = 1
    elif player1 == "papel" and player2 == "tijeras":
        result = 2
    return result

resultado=search_winner("papel", "tijeras")
print("Gana el jugador ",resultado)

test = [
    ["piedra", "piedra", 0],
    ["piedra", "tijeras", 1],
    ["piedra", "papel", 2]
]

for partida in test:
    print ("Player 1: %s player 2: %s Winner: %s Validation: %s"
           % (
                partida[0],
                partida[1],
                search_winner(partida[0], partida[1]),
                partida[2])
           )
    
#Eleccion de player 1 de forma aleatria
def get_choice():
    return choice(options)

for i in range(10):
    player1 = get_choice()
    player2 = get_choice()
    print ("Player 1: %s Player 2: %s Winner: %s"
           % (
                player1,
                player2,
                search_winner(player1, player2)
                )
           )
    
#Inicia la red neuronal
def str_to_list(option):
    if option == "piedra":
        res = [1, 0, 0]
    elif option == "tijeras":
        res = [0, 1, 0]
    else:
        res = [0, 0, 1]
    return res

data_X = list(map(str_to_list, ["piedra", "tijeras", "papel"]))
data_y = list(map(str_to_list, ["pLapel", "piedra", "tijeras"]))

print(data_X)
print(data_y)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(verbose=False, warm_start=True)

model = clf.fit([data_X[0]], [data_y[0]])
print(model)

def play_and_learn(iterations=10, debug=False):
    score = {"win": 0, "loose": 0}
    data_X = []
    data_y = []
    
    for i in range(iterations):
        player1 = get_choice()
        
        predict = model.predict_proba([str_to_list(player1)])[0]
        
        if predict[0] >= 0.95:
            player2 = options[0]
        elif predict[1] >= 0.95:
            player2 = options[1]
        elif predict[2] >= 0.95:
            player2 = options[2]
        else:
            player2 = get_choice()
            
        if debug == True:
            print("Player 1: %s Player 2 (Modelo): %s --> %s" % (player1, predict, player2))
            
        winner = search_winner(player1, player2)
        if debug == True:
            print("Comprobamos: Player 1 VS Player 2: %s" % winner)
            
        if winner == 2:
            data_X.append(str_to_list(player1))
            data_y.append(str_to_list(player2))
            score["win"] += 1  
        else:
            score["loose"] += 1
    return score, data_X, data_y

score, data_X, data_y = play_and_learn(1, debug=False)
print(data_X)
print(data_y)
print("Score %s %s %%" % (score, (score["win"] * 100 / (score["win"] + score["loose"]))))
if len(data_X):
    model = model.partial_fit(data_X, data_y)
    
i = 0
historic_pct = []
while True:
    i += 1
    score, data_X, data_y = play_and_learn(1000, debug=False)
    pct = (score["win"] * 100 / (score["win"] + score["loose"]))
    historic_pct.append(pct)
    print("Iteraction %s - Score %s %s %%" % (i, score, pct))
    if len(data_X):
        model = model.partial_fit(data_X, data_y)
    if sum(historic_pct[-9:]) == 900:
        break
    
import matplotlib.pyplot as plt

x = list(range(len(historic_pct)))
y = historic_pct

fig, ax = plt.subplots()
ax.set_ylabel("%")
ax.set_xlabel("Iteracion")
ax.set_title("Porcentaje de aprendizaje en cada iteracion")
ax.plot(x, y)
plt.show()
