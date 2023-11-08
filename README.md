# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) DSI #39 Capstone Project: Productivity State Detector (PSD)

### Problem Statement
In today's world of social media applications, the main currency is your attention. Major companies are vying for people's attention in exchange for entertainment or convenience.

However this does comes at a cost of shorter attention spans. Research have shown that the average attention spans have shorten. And in turn, when people are less focused on their task, they may be more error prone or may have difficult processing their tasks.

On the other end of the spectrum, there may people too engrossed in their tasks that they ignore signs of fatigue.

---

### Project Objective

This project aims to develop an application to:
1. Detect productivity 
2. Detect signs of fatigue
3. Help users overcome their distractions 

---

### Definitions

Within the scope of this project:
- "Productive" is defined as the user looking at the centre of the screen.
- "Fatigue" is the user closing their eyes and opening their mouth (i.e. yawning)

---
### Data Used

Data is recorded using a webcam at 720p (1280x720), 16:9, 30fps, mp4 format.

---

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**x, y, z**|*float*|Coordinates of the facial and body landmarks extracted using mediapipe|
|**productivity**|*string*| Whether subject is productive|
|**fatigue**|*string*|Whether subject is showing signs of fatigue|
|**productivity_probability**|*float*|Probability of a single frame that subject is productive or not|
|**fatigue_probability**|*int64*|Probability of a single frame that subject is showing signs of fatigue or not|

---

### Notebook description

* [01_train](/code/01_train.ipynb): Extracing the facial and body landmarks, converting them into coordinates and training a classification model
* [02_test](/code/02_test.ipynb): Testing the model and building the chatbot
* [03_deploy_local](/code/03_deploy_local_nokey.ipynb): Deploying the model and chatbot on local Streamlit
* [03_deploy_cloud](/code/03_deploy_cloud.ipynb): Deploying the model and chatbot on cloud Streamlit

---

### Conclusion

- A classification model with good accuracy (> 0.9) and good cross validation score was built to identify whether an user is productive or showing signs of fatigue via video recording.
- An user-friendly web application (https://laser-focus.streamlit.app/) was also built to deploy this model to users, with the addition of a chatbot to provide users tips on managing productivity and their fatigue based on their use case.

---

### Future Work

#### 1. Alternative Use Cases:
a) Employee Monitoring: Checking if employees working in non-physical workspaces are productive (e.g. employees who are working from home).

b) Employee Welfare Management: Checking for signs of burn-out in employees (e.g. incorporating this into the facial recognition system used to manage access to the office)
​
#### 2. Refine the "productive" model to consider more practical use cases such as if the user has a secondary screen.​

#### 3. Refine the "fatigue" model to consider more signs of fatigue such as lowering of head. 

---

