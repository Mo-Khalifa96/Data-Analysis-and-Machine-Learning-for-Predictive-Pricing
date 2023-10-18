# Data Analysis & Machine Learning for Predictive Pricing 

## About The Project 
**This project utilizes Python for data analysis and machine learning. It covers core data science aspects from exploratory data analysis and data 
wrangling to advanced statistical analysis to employing machine learning algorithms for predictive pricing. The project was originally completed as 
part of my IBM course, 'Data Analysis with Python', but was expanded and built upon to cover a wider range of data science methods and skills learned 
within and outside the course.**
<br>
<br>
**The aim of this project is to develop a machine learning model that can reliably predict car prices. To develop such a model, an automobile dataset, comprised of
different car attributes and capabilities and the actual prices corresponding to each set of attributes, is fed to the model to train it. More particularly, the 
dataset is filtered, statistically analyzed, and only the most relevant attributes are selected to train the model. To obtain the best performing model, the one
most capable of generating reliable predictions, different models are developed, fine-tuned, and evaluated through in-sample and out-of-sample evaluations, and 
their performances compared. The final end is to find the model that simultaneously performs best on the data by which it was trained (in-sample) and in the real 
world with novel or previously unseen data (out-of-sample). As such, the model undergoes a process of parameter fine-tuning to reduce its estimated generalization 
error and thereby improve its overall performance in the real world. The model selected is then used to generate price predictions both from a data sample and from 
user input.** <br>

<br>

**Overall, the project is broken down into six sections: <br>
&emsp; 1) Reading and Inspecting Data <br>
&emsp; 2) Updating and Cleaning Data <br>
&emsp; 3) Data Selection and Preprocessing <br>
&emsp; 4) Model Development and Evaluation <br>
&emsp; 5) Hyperparameter Tuning <br>
&emsp; 6) Model Prediction** <br>

<br>
<br>

## About The Data 
**As mentioned, the data being analyzed here is based on an automobile dataset comprised of a variety of car characteristics, including key characteristics such as car brand, 
horsepower, engine type, and its original pricing. The automobile dataset can be accessed from the attached Excel file or by clicking [here](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data).**
<br>

**You can view each coloumn in the set and its description in the table below:** <br>
<br>

| **Variable**          | **Description**                                                                                  |
| :-----------------    | :------------------------------------------------------------------------------------------------|
| **symboling**         | Car's insurance risk level (continuous from -3 to 3).                                            |
| **normalized-losses** | Relative average loss payment per insured vehicle year (continuous from 65 to 256).              |
| **make**              | Car's brand or manufacturer name (includes, alfa-romero, audi, bmw, chevrolet, dodge, honda,isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo).|
| **fuel-type**         | Car's fuel type (diesel or gas).                                                                 |
| **aspiration**        | Car's aspiration engine type (std or turbo).                                                     |
| **num-of-doors**      | Number of doors (two or four)                                                                    |
| **body-style**        | Car's body style (hardtop, wagon, sedan, hatchback, convertible)                                 |
| **drive-wheels**      | Type of driving wheels (4wd, fwd, rwd).                                                          |
| **engine-location**   | Car's engine location (front or rear).                                                           |
| **wheel-base**        | Car's wheelbase distance (continuous from 86.6 to 120.9)                                         |
| **length**            | Car's length (continuous from 141.1 to 208.1).                                                   |
| **width**             | Car's width (continuous from 60.3 to 72.3).                                                      |
| **height**            | Car's height (continuous from 47.8 to 59.8).                                                     |
| **curb-weight**       | Car's curb weight (continuous from 1488 to 4066).                                                |
| **engine-type**       | The engine type (includes, dohc, dohcv, ohc, ohcf, ohcv, l, rotor).                              |
| **num-of-cylinders**  | Number of cylinders (two, three, four, five, six, eight, twelve).                                |
| **engine-size**       | Car's engine size (continuous from 61 to 326).                                                   |
| **fuel-system**       | Car's fuel system (includes, 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi).                      |
| **bore**              | Car's bore size (continuous from 2.54 to 3.94).                                                  |
| **stroke**            | Engine's stroke length (continuous from 2.07 to 4.17).                                           |
| **compression-ratio** | Ratio between the cylinder's volume and combustion chamber in combustion engine (continuous from 7 to 23).|
| **horsepower**        | Car's horsepower (continuous from 48 to 288).                                                     |
| **peak-rpm**          | Peak revolutions per minute (continuous from 4150 to 6600).                                       |
| **city-mpg**          | Car's average miles per gallon in the city (continuous from 13 to 49).                            |
| **highway-mpg**       | Car's average miles per gallon on highways (continuous from 16 to 54).                            |
| **price**             | Car price (continuous from 5118 to 45400).                                                        |

<br>
<br>

**Here's a sample of the dataset being analyzed:**
<br> 

<img src="automobile data screenshot.jpg" alt="https://github.com/Mo-Khalifa96/Data-Analysis-and-Machine-Learning-for-Predictive-Pricing/blob/main/automobile%20data%20screenshot.jpg" width="800"/>

<br>
<br> 

## Quick Access 
**As always, I have provided two links to quickly access the project. Both will direct you to the Jupyter Notebook with all the code and corresponding output, broken down and 
organized into separate sections or cells, each section supplied with explanations and takeaways that guide the development of the project one step at a time. However, whilst 
the first link renders the code and output for viewing only, the second one will enable you in addition to interact with the code and reproduce the results if you wish so. I would 
recommend the second link as the final section provides a function that takes input from the user with all the car attributes they have in mind, and employs the model to return back 
a price prediction that best corresponds to these given attributes. Feel free to try it yourself.** <br> 
**To execute the code, please make sure to run the first two cells first in order to install and be able to use the Python packages for performing the necessary analyses. To run any 
given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.**
<br>
<br>
<br>
***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Data-Analysis-and-Machine-Learning-for-Predictive-Pricing/blob/main/Data%20Analysis%20%26%20Machine%20Learning%20for%20Predictive%20Pricing.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Data-Analysis-and-Machine-Learning-for-Predictive-Pricing/main?labpath=Data%20Analysis%20%26%20Machine%20Learning%20for%20Predictive%20Pricing.ipynb
<br>
<br>


