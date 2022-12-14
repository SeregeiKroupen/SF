# Description of module_0 projects. 
   
There are two projects in module one of DST course:  
1. Number guessing script || *"rundom_game.py"* and *"Real-Random.py"*
2. Tik-Tac-Toe game || *"tic-tac-toe2.py"*. 
---

### Number guessing script  

Random guessing takes 101 attempts to guess a number in the range from 1 to 100. Using knowledge of whether the "guess" is more or 
less than the number - takes 30 attetemps. **The main task is to guess a number using less than 30 attempts on average.**   

I used [binary search algorithm](https://en.wikipedia.org/wiki/Binary_search_algorithm).
This method moves the results down to less than 6 attempts (*"rundom_game.py"*).    

But, I have in mind that random numbers have been generated using method `np.random.randint`, which creates not "actually" random numbers.
So I collected the numbers from purchasing cheques connecting to the product items. It was not a very big collection. 
But the script made an even better result - 5.6 attempts (*"Real-Random.py"*).   

---
### "Tik-Tac-Toe" game   
"tic-tac-toe2.py"     
   
It is the script of [the game](https://en.wikipedia.org/wiki/Tic-tac-toe) to play on a console without any GUI.
The task was to code just game-play. But I made two more features. 
First, I add an "AI" script, which is a "short winning strategy algorithm" for the game. So you can play the game like with the computer. 
Second, I add "easter egg". It is a joke. If the player answer in the wrong way wh- the playing board updating 
with random marks for both players. Try to realize, where you have to go next time!
   
---
PS. These are my first two projects on Python.
Used basic algorithm statments, basic operators and algorithmic constructions, sach control flow, imput/output and etc.
