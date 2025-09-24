Al-Driven Crop Planner is a smart agricultural advisory platform built using machine leaming. 
It is designed to help farmers in Telangana make informed crop cultivation decisions by analyzing soil and environmental inputs.
The system accepts user inputs like soil pH, soil type, water availability, and district name.
It then leverages an ML.
Model trained on real agricultural data to recommend the top five crops best suited for the given conditions.
In addition to recommendations, the system provides a comprehensive crop planning module for each suggested crop.
This includes agronomic guidance, cost estimation, and cultivation timelines.
To enhance accessibility and usability, an integrated chatbot assists users with queries related to crops, farming practices, and planning details.

System Architecture and Workflow:
  User Interface (Frontend): Collects input values soil pH, soil type, water availability, district.
  Backend Processing:Inputs are passed to a trained ML. model (RandomForestClassifier).
  Model predicts and ranks crops based on suitability.
  Outputs are returned to the frontend.
  Crop Detail Module: Users can click on a crop to view detailed guidelines.
  Chatbot: Accessible on home page to provide crop information.
