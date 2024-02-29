

options = ["piedra", "papel", "tijera"]

#Generamos una funcion con todas las posibles combinaciones de jugadas
def search_winner(p1, p2):
    if p1 == p2:
        result = 0
    elif p1 == "piedra" and p2 == "tijera":
        result = 1
    elif p1 == "piedra" and p2 == "papel":
        result = 2
    elif p1 == "tijera" and p2 == "piedra":
        result = 2
    elif p1 == "tijera" and p2 == "papel":
        result = 1
    elif p1 == "papel" and p2 == "piedra":
        result = 1
    elif p1 == "papel" and p2 == "tijera":
        result = 2
    return result

print(search_winner("piedra", "tijera"))

test = [
    ["piedra", "piedra", 0],
    ["piedra", "tijeras", 1],
    ["piedra", "papel", 2],
]

for partida in test:
    print("player 1: %s player 2: %s Winner: %s Validation: %s" 
    % (
        partida[0], 
        partida[1], 
        search_winner(partida[0], partida[1]), 
        partida[2])
    )
    
from random import choice

def get_choice():
    return choice(options)

for i in range(10):
    player1 = get_choice()
    player2 = get_choice()
    print("player 1: %s player 2: %s Winner: %s" % (player1, player2, search_winner(player1, player2)))
    
#inicia la red neuronal 