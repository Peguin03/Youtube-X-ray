import streamlit as st
import cv2
from PIL import Image
from pytube import YouTube
import clip
import math
import numpy as np
import torch
import plotly.express as px
import datetime
import time
# import base64

# main_bg = "net.webp"
# main_bg_ext = "webp"

# st.markdown(
#     f"markdown""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title('Youtube X-ray')
'# YouTube X-ray'
# '# Watch what you want.'
st.subheader('Watch what you **want**.')

video_frames = []
N = 120
# st.write('### Paste youtube URL here:')
video_url = st.text_input('Paste youtube URL here:')
search = st.text_input('What would you like to search?')

@st.cache
def download(streams):
  print("Downloading...")
  streams[0].download(filename="video.mp4")
  print("Download completed.")


def process():

  text = st.empty()
  st.text('Searching...')
  bar = st.progress(0)
  # st.info('Searching..')
  # st.text('Downloading Video...')
  streams = YouTube(video_url).streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)

  # Check if there is a valid stream
  if len(streams) == 0:
    st.write("No suitable video found for this YouTube video!")
    raise "No suitable video found for this YouTube video!"

  # Download the video as video.mp4
  download(streams)
  # st.text('Video Downloaded...')
  bar.progress(20)

  capture = cv2.VideoCapture('video.mp4')
  fps = capture.get(cv2.CAP_PROP_FPS)
  current_frame = 0

  # st.text('Extracting Frames...')
  bar.progress(30)
  while capture.isOpened():
    # Read the current frame
    ret, frame = capture.read()

    # Convert it to a PIL image (required for CLIP) and store it
    if ret == True:
      video_frames.append(Image.fromarray(frame[:, :, ::-1]))
    else:
      break

    # Skip N frames
    current_frame += N
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

  # Print some statistics
  l = len(video_frames)
  print(f"Frames extracted: {l}")
  # st.text(str(l)+' Frames extracted.')
  bar.progress(40)

  # st.text('Loading Model...')
  bar.progress(50)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)

  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)

  # The encoded features will bs stored in video_features
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

  # st.text('Preprocessing Images...')
  bar.progress(60)

  # Process each batch
  for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    
    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    
    # Encode with CLIP and normalize
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)

    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))

  # Print some stats
  print(f"Features: {video_features.shape}")

  def search_video(search_query, display_heatmap=False, display_results_count=3):

    # Encode and normalize the search query using CLIP
    with torch.no_grad():
      text_features = model.encode_text(clip.tokenize(search_query).to(device))
      text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the similarity between the search query and each frame using the Cosine similarity
    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)

    # Display the heatmap
    if display_heatmap:
      print("Search query heatmap over the frames of the video:")
      fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis')
      fig.update_layout(coloraxis_showscale=False)
      fig.update_xaxes(showticklabels=False)
      fig.update_yaxes(showticklabels=False)
      fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
      fig.show()
      # print()

    # st.text('Success!')
    bar.progress(100)
    st.success("Here are the top 3 Results!")
    # Display the top 3 frames
    for frame_id in best_photo_idx:
      # display(video_frames[frame_id])
      st.image(video_frames[frame_id])
      # Find the timestamp in the video and display it
      seconds = round(frame_id.cpu().numpy()[0] * N / fps)

      st.markdown(str(video_url)+'&t='+str(math.floor(seconds)) , unsafe_allow_html=True)

  # st.text('Searching the Video...')
  bar.progress(80)
  search_video(search)

if st.button('Go!'):
  process()