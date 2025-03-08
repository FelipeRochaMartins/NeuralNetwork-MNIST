import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class DrawingPredictor:
    """
    A graphical interface for drawing digits and making predictions using a trained neural network.
    
    This class creates a window with a canvas where users can draw digits using their mouse.
    The drawing is then processed and fed into a neural network for prediction.
    
    Features:
        - 280x280 drawing canvas (automatically scaled to 28x28 for MNIST)
        - Real-time probability visualization for each digit
        - Clear button to reset the canvas
        - Predict button to run neural network inference
    
    Attributes:
        network: The trained neural network model for making predictions
        root: The main Tkinter window
        canvas: The drawing canvas widget
        image: PIL Image object for storing the drawing
        draw: PIL ImageDraw object for drawing
        result_label: Label widget showing the prediction result
        prob_bars: List of progress bars showing probabilities for each digit
        prob_labels: List of labels showing probability percentages
    """
    
    def __init__(self, network):
        """
        Initialize the DrawingPredictor interface.
        
        Args:
            network: A trained neural network model that implements predict() method
                    The model should accept 784-dimensional input (28x28 flattened)
                    and output 10-dimensional probabilities for digits 0-9
        """
        self.network = network
        self.root = tk.Tk()
        self.root.title("Draw a Digit")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='black')
        self.canvas.pack(pady=20)
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create an image to draw on
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result label
        self.result_label = ttk.Label(self.root, text="Draw a digit and click Predict")
        self.result_label.pack(pady=10)
        
        # Probability bars frame
        self.prob_frame = ttk.Frame(self.root)
        self.prob_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            label = ttk.Label(self.prob_frame, text=str(i))
            label.grid(row=i, column=0, padx=5)
            
            progress = ttk.Progressbar(self.prob_frame, length=200, mode='determinate')
            progress.grid(row=i, column=1, padx=5)
            
            value_label = ttk.Label(self.prob_frame, text="0%")
            value_label.grid(row=i, column=2, padx=5)
            
            self.prob_bars.append(progress)
            self.prob_labels.append(value_label)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
    
    def start_drawing(self, event):
        """
        Handle mouse button press event to start drawing.
        
        Args:
            event: Tkinter event object containing mouse coordinates
        """
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        """
        Handle mouse motion event to draw lines.
        
        Creates smooth lines by connecting the current mouse position
        to the last known position with a white line.
        
        Args:
            event: Tkinter event object containing mouse coordinates
        """
        if self.drawing:
            x, y = event.x, event.y
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  fill='white', width=20, capstyle=tk.ROUND, 
                                  smooth=tk.TRUE)
            # Draw on image
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill='white', width=20)
            self.last_x = x
            self.last_y = y
    
    def stop_drawing(self, event):
        """
        Handle mouse button release event to stop drawing.
        
        Args:
            event: Tkinter event object containing mouse coordinates
        """
        self.drawing = False
    
    def clear(self):
        """
        Clear the drawing canvas and reset all predictions.
        
        This method:
        - Clears the canvas
        - Creates a new blank image
        - Resets the prediction label
        - Resets all probability bars to 0
        """
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")
        for i in range(10):
            self.prob_bars[i]['value'] = 0
            self.prob_labels[i]['text'] = "0%"
    
    def predict(self):
        """
        Process the drawn digit and make a prediction.
        
        This method:
        1. Resizes the drawing to 28x28 pixels
        2. Normalizes pixel values to range [0, 1]
        3. Feeds the processed image through the neural network
        4. Updates the UI with predictions and probabilities
        """
        # Resize to 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized)
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for the network
        img_array = img_array.reshape(1, 784)
        
        # Get predictions
        probabilities = self.network.predict(img_array, return_probabilities=True)
        prediction = np.argmax(probabilities[0])
        
        # Update result label
        self.result_label.config(text=f"Predicted digit: {prediction}")
        
        # Update probability bars
        for i in range(10):
            prob = probabilities[0][i] * 100
            self.prob_bars[i]['value'] = prob
            self.prob_labels[i]['text'] = f"{prob:.1f}%"
    
    def run(self):
        """
        Start the drawing predictor application.
        
        This method starts the Tkinter event loop and displays the window.
        The application will run until the window is closed.
        """
        self.root.mainloop()

def launch_drawing_predictor(network):
    """
    Create and launch a DrawingPredictor instance.
    
    This is a convenience function to create and run the drawing predictor
    in a single line of code.
    
    Args:
        network: A trained neural network model for digit recognition
    """
    app = DrawingPredictor(network)
    app.run()
