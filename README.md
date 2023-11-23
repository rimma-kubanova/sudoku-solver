## SudokuSolverAI

SudokuSolverAI - an app for solving sudoku puzzles through a web camera (augmented reality). 

The program detects the sudoku puzzle using the webcam and implements OpenCV for image processing. Using a neural network for digit prediction, it solves the puzzle using an efficient backtracking algorithm. If a solution exists, it showcases the answer on the screen.

## Demonstration

Screenshots of solving Sudoku through MacBook Webcam:

<img height="450" alt="Screenshot 2023-09-08 at 14 54 27" src="https://github.com/rimma-kubanova/sudoku-solver/assets/115300909/ebcbbc66-3e37-4a7d-9e65-576991ca0c49">
<img height="450" alt="Screenshot 2023-09-08 at 14 54 27" src="https://github.com/rimma-kubanova/sudoku-solver/assets/115300909/07e4b142-51b1-4709-8d2f-491b04a59f70">

## Process

Here is the step-by-step how image processing and image detection works in the program using OpenCV and Tensorflow.
<img height="300" alt="Screenshot 2023-09-08 at 14 54 27" src="https://github.com/rimma-kubanova/sudoku-solver/assets/115300909/d5a9c6d5-01db-451f-a4c1-5e90022e7a58">
<img height="300" alt="Screenshot 2023-09-08 at 15 14 40" src="https://github.com/rimma-kubanova/sudoku-solver/assets/115300909/c53d5d04-e233-434e-96e5-dc328f089f09">
<img height="300" alt="Screenshot 2023-11-23 at 18 01 14" src="https://github.com/rimma-kubanova/sudoku-solver/assets/115300909/5ce4233a-e4e1-48a1-b737-61da82d012e9">

* 1 step: Process the image using the gaussian blur, threshold filter to get the grids of sudoku more accurate.
* 2 step: Find corners and crop the image by outer corners.
* 3 step: Detect the digits using the trained model. Detect the grids and identify order of numbers
* 4 step: Use the backtracking algorithm to solve the sudoku.
* 5 step: Show the solution on the empty grids of the sudoku.
## Setup

The project is created with:
* Python: 3.11.5
* OpenCV: 4.8.0
* NumPy: 1.24.3
* Tensorflow: 2.13.0
* Keras: 2.13.1
  
To run this project, first install all libraries mentioned in requirements.txt using the code below:

```
pip install -r requirements.txt
```
Then run the project in the main package using the following code:
```
python3 main.py
```
