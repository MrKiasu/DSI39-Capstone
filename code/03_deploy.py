import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os

st.set_option('deprecation.showPyplotGlobalUse', False) #ignore warnings

# Load the Producitivity pickle
with open("productive.pkl", 'rb') as f:
    model_prod = pickle.load(f)

# Load the Producitivity pickle
with open("fatigue.pkl", 'rb') as f:
    model_fat = pickle.load(f)

mp_holistic = mp.solutions.holistic

st.image("bruce_lee_laser_focus_quote.jpg", use_column_width=True)

st.title("Productivity State Developer (PSD)")
st.subheader("The application is developed to help you understand your productivity patterns and also provide a solution for any productivity issues.")
st.markdown("Disclaimer: This application is a student project and may not represent a production-quality solution. It is being developed for educational purposes, and its functionality, accuracy, and performance may be limited. Please use the app with the understanding that it is a work in progress.")

st.divider()

st.write("Step 1: Record yourself doing a task on your laptop/PC via the webcam.")

st.divider()

st.write("Step 2: Upload your video file for processing.")

# File upload widget
uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_video is not None:
    st.success("Video is uploaded!")

button_process = st.button("Process Video", key="button_process")

# if button is pressed
if button_process:
    with st.spinner("Processing..."):    # [KIV] Using a custom, animated spinner

        temp_video_path = "temp.mp4" #tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_video.read())

        # output_video_path = 'videos/output.mp4'#tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        # annotated_video_path = annotate_video(temp_video_path, output_video_path, model)

        # with open(annotated_video_path, "rb") as video_file:
        # video_bytes = video_file.read()
        # st.write(annotated_video_path)
        # st.video(video_bytes)   
        
        # # Create an auto-download button
        # st.download_button(
        #     label="Download File",
        #     data=video_bytes,
        #     file_name='Annotated Video.mp4',

        # Remove the temporary files (only after all the previous code has completed running)
        # os.remove(output_video_path)

        # Open webcam
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # Frames per Second
        
        if "fps" not in st.session_state:
            st.session_state["fps"] = fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        my_bar = st.progress(0, text="Processing Video File...")
        frame_counter = 0

        if os.path.isfile("coords.csv"):  # delete existing file if found
            os.remove("coords.csv")
        else:
            pass


        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()

                # [TEST] For pre-recorded video
                if not ret:
                    break  # If there are no more frames to read, break out of the loop
                
                frame_counter = frame_counter + 1
                my_bar.progress(frame_counter/total_frames, text="Processing Video File...")

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Concatenate rows
                    row = pose_row + face_row
                        
                    # Export to CSV
                    with open("coords.csv", mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
                
                except:
                    pass

                # press q key to terminate webcam capture mode
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    
                    break

        # out.release()
        cap.release()
        cv2.destroyAllWindows()
        
        os.remove(temp_video_path)  # delete temp file.

        st.success("Video is processed!")

st.divider()

st.write("Step 3: Analyze your productivity state.")

button_analyse = st.button("Analyse!", key="button_analyse")

# if button is pressed
if button_analyse:
    with st.spinner("Analysing..."):    # [KIV] Using a custom, animated spinner
        
        fps = st.session_state["fps"]

        df = pd.read_csv("coords.csv")

        prod_list = []
        fat_list = []
        prod_prob_list = []
        fat_prob_list = []

        analyse_bar = st.progress(0, text="Analysing...")

        total_rows = len(df.index)
        row_counter = 0

        for i in df.index:
            prod_list.append(model_prod.predict(df.iloc[i].values.reshape(1,-1))[0])
            fat_list.append(model_fat.predict(df.iloc[i].values.reshape(1,-1))[0])
            
            row_counter = row_counter + 1
            analyse_bar.progress(row_counter / total_rows, text="Analysing...")

            prod_prob_list.append(round(model_prod.predict_proba(df.iloc[i].values.reshape(1,-1))[0].max(),2))
            fat_prob_list.append(round(model_fat.predict_proba(df.iloc[i].values.reshape(1,-1))[0].max(),2))

        df = pd.DataFrame({"Productive": prod_list, "Fatigue": fat_list, "Productive Probability": prod_prob_list, "Fatigue Probability": fat_prob_list })

        df["Productive"].replace("Productive", 1, inplace = True)
        df["Productive"].replace("Not Productive", 0, inplace = True)
        df["Fatigue"].replace("Fatigue", 1, inplace = True)
        df["Fatigue"].replace("Not Fatigue", 0, inplace = True)


        # Create a line plot using Seaborn
        sns.set(style="whitegrid") 
        ax = sns.lineplot(x=df.index, y=df["Productive"])

        plt.xlabel('Time (seconds)')
        plt.title('Productivity')
        plt.ylim(-0.1, 1.1) # the axis limits are set to -0.1 to 1.1 instead of the default 0 to 1 to show the lines clearer

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not Productive","Productive"])

        # Display the chart
        st.pyplot()

        sns.set(style="whitegrid") 
        ax = sns.lineplot(x=df.index, y=df["Fatigue"])

        plt.xlabel('Time (seconds)')
        plt.title('Fatigue')
        plt.ylim(-0.1, 1.1) # the axis limits are set to -0.1 to 1.1 instead of the default 0 to 1 to show the lines clearer

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not Fatigue","Fatigue"])
        
        # Display the chart
        st.pyplot()

        prod_percent = sum(df["Productive"]) / len(df.index) * 100
        fat_percent = sum(df["Fatigue"]) / len(df.index) * 100
        elapsed_time = len(df.index) / fps

        st.write(f"Out of {elapsed_time:.1f} seconds, you were productive for {sum(df['Productive'])/fps:.1f} seconds ({prod_percent:.0f}%) and showed signs of fatigue for {sum(df['Fatigue'])/fps:.1f} seconds ({fat_percent:.0f}%)")


    st.success("Done!")

st.divider()

st.write("Step 4: If required, seek help from our friendly chatbot!")
st.write("Sample prompt: How do I improve my productivity and stop doomscrolling?")

user_content = st.text_area("What would you like to talk about?")

button_chat = st.button("Ask!", key="button_chat")

# if button is pressed
if button_chat:
    with st.spinner("Hmmm..."):    # [KIV] Using a custom, animated spinner
        
        from langchain.chat_models import ChatOpenAI
        from langchain.schema.messages import HumanMessage, SystemMessage
        os.environ['OPENAI_API_KEY'] = "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" # replace with your API key

        chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.1)

        messages = [
            SystemMessage(
                content="You are a friendly and helpful assistant tasked to answer questions about improving oneself."
            ),
            HumanMessage(
                content=user_content
            ),
                    ]

        response = chat(messages).content
   
        st.success(response)

st.divider()