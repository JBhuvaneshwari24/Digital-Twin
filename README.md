# Digital-Twin
Project Overview

This project is a Smart Health Assessment System that analyzes a person’s overall health using three important factors:

Physical Activity, Sleep, Stress

Instead of analyzing each separately, the system combines all three and generates a single Health Index (0–100) to represent overall health.
# Objective

The main goal of this project is to:

Build separate machine learning models for stress, sleep, and activity

Combine them using multimodal fusion

Provide a real-time dashboard for user input

Generate health score + recommendations
# Features
 Multimodal health analysis
 
 Real-time user input via dashboard
 
 Individual risk prediction (Stress, Sleep, Activity)
 
 Final Health Index (0–100)
 
 Personalized health recommendations
 
 Interactive visual dashboard
 # System Workflow
 
User enters: Steps, Sleep hours, Heart rate

Data is preprocessed

Each model predicts risk: Activity Model, Sleep Model, Stress Model

Fusion Engine combines results

Final Health Index is generated
# Datasets Used
WESAD Dataset → Stress detection

Sleep-EDF Dataset → Sleep analysis

Fitbit Dataset → Physical activity
# Output

The system provides:

Health Index (0–100)

Risk Levels:
Stress Risk, Sleep Risk, Activity Risk

Health Category: Good, Moderate, High Risk

Personalized Recommendations
# Models Used
Stress Model → Random Forest

Sleep Model → Random Forest

Activity Model → MLP (Neural Network)
# Fusion Technique

Outputs are converted to scores

Normalized to same scale

Combined using averaging fusion
# Advantages
Combines multiple health factors

Provides simple and clear output

Real-time prediction

Easy to use dashboard

More accurate than single-model systems
# Conclusion

This system provides a complete and practical health monitoring solution by combining stress, sleep, and activity into a single Health Index. It helps users easily understand their health and improve their lifestyle.

# Author

Bhuvaneshwari, Deepika
